/* 2048 bitboard engine, n-tuple network with TD(0)/TC learning, expectimax search.
 *
 * Board: 128-bit integer, cell i (row-major, 0..15) lives in bits 5i..5i+4 and
 * holds the tile exponent (0 = empty, up to 17 = 131072, the largest tile a 4x4
 * board can hold). Rules match game2048/core.py.
 *
 * Directions: 0 = up, 1 = right, 2 = down, 3 = left.
 */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef __uint128_t board_t;
typedef uint32_t row_t;

#define CB 5                     /* bits per cell */
#define CMASK 31
#define MAX_EXP 17
#define RADIX 18                 /* table digit per cell: exponents 0..17 */
#define ROW_BITS 20
#define ROW_MASK 0xFFFFFu
#define ROW_N (1u << ROW_BITS)
#define MAX_PATTERNS 16
#define MAX_LEN 8
#define MAX_STAGES 8
#define TT_BITS 20

static inline int cell(board_t b, int i) { return (int)((b >> (CB * i)) & CMASK); }
static inline board_t put(int v, int i) { return (board_t)v << (CB * i); }
static inline board_t mk(uint64_t lo, uint64_t hi) { return ((board_t)hi << 64) | lo; }
static inline uint64_t lo64(board_t b) { return (uint64_t)b; }
static inline uint64_t hi64(board_t b) { return (uint64_t)(b >> 64); }
static inline uint64_t hash128(board_t b) {
    return (lo64(b) * 0x9E3779B97F4A7C15ULL) ^ (hi64(b) * 0xD1B54A32D192ED03ULL + 0x632BE59BD9B4E019ULL);
}
static inline size_t tt_hash(board_t b) { return (size_t)(hash128(b) >> (64 - TT_BITS)); }

/* ---------------------------------------------------------------- engine --- */

static row_t *row_left, *row_right;
static int32_t *row_reward_l, *row_reward_r;
static int tables_ready = 0;
static pthread_mutex_t tables_lock = PTHREAD_MUTEX_INITIALIZER;

static inline row_t reverse_row(row_t r) {
    return ((r >> 15) & 0x1F) | (((r >> 10) & 0x1F) << 5) | (((r >> 5) & 0x1F) << 10) | ((r & 0x1F) << 15);
}

static void init_tables(void) {
    pthread_mutex_lock(&tables_lock);
    if (!tables_ready) {
        row_left = (row_t *)malloc(sizeof(row_t) * ROW_N);
        row_right = (row_t *)malloc(sizeof(row_t) * ROW_N);
        row_reward_l = (int32_t *)malloc(sizeof(int32_t) * ROW_N);
        row_reward_r = (int32_t *)malloc(sizeof(int32_t) * ROW_N);
        for (uint32_t idx = 0; idx < ROW_N; idx++) {
            int cells[4] = {(int)(idx & CMASK), (int)((idx >> 5) & CMASK), (int)((idx >> 10) & CMASK), (int)((idx >> 15) & CMASK)};
            int tiles[4], nt = 0;
            for (int i = 0; i < 4; i++) if (cells[i]) tiles[nt++] = cells[i];
            int out[4] = {0, 0, 0, 0}, no = 0, reward = 0;
            for (int i = 0; i < nt;) {
                if (i + 1 < nt && tiles[i] == tiles[i + 1] && tiles[i] < MAX_EXP) {
                    out[no++] = tiles[i] + 1;
                    reward += 1 << (tiles[i] + 1);
                    i += 2;
                } else {
                    out[no++] = tiles[i++];
                }
            }
            row_t res = (row_t)(out[0] | (out[1] << 5) | (out[2] << 10) | (out[3] << 15));
            row_left[idx] = res;
            row_reward_l[idx] = reward;
            row_right[reverse_row(idx)] = reverse_row(res);
            row_reward_r[reverse_row(idx)] = reward;
        }
        tables_ready = 1;
    }
    pthread_mutex_unlock(&tables_lock);
}

static inline board_t transpose(board_t x) {
    board_t out = 0;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) out |= put(cell(x, r * 4 + c), c * 4 + r);
    return out;
}

static inline board_t slide_left(board_t b, int *reward) {
    board_t r = 0;
    int s = 0;
    for (int i = 0; i < 4; i++) {
        row_t row = (row_t)((b >> (ROW_BITS * i)) & ROW_MASK);
        r |= (board_t)row_left[row] << (ROW_BITS * i);
        s += row_reward_l[row];
    }
    *reward = s;
    return r;
}

static inline board_t slide_right(board_t b, int *reward) {
    board_t r = 0;
    int s = 0;
    for (int i = 0; i < 4; i++) {
        row_t row = (row_t)((b >> (ROW_BITS * i)) & ROW_MASK);
        r |= (board_t)row_right[row] << (ROW_BITS * i);
        s += row_reward_r[row];
    }
    *reward = s;
    return r;
}

static inline board_t do_move(board_t b, int dir, int *reward) {
    switch (dir) {
        case 0: return transpose(slide_left(transpose(b), reward));
        case 1: return slide_right(b, reward);
        case 2: return transpose(slide_right(transpose(b), reward));
        default: return slide_left(b, reward);
    }
}

static inline int count_empty(board_t b) {
    int n = 0;
    for (int i = 0; i < 16; i++) n += cell(b, i) == 0;
    return n;
}

static inline int max_exp(board_t b) {
    int m = 0;
    for (int i = 0; i < 16; i++) {
        int v = cell(b, i);
        if (v > m) m = v;
    }
    return m;
}

static inline uint64_t rng_next(uint64_t *s) {  /* splitmix64 */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline board_t spawn(board_t b, uint64_t *rng) {
    int empty = count_empty(b);
    if (!empty) return b;
    uint64_t r = rng_next(rng);
    int k = (int)((r & 0xFFFFFFFFULL) % (uint64_t)empty);
    int val = ((r >> 32) % 10 == 0) ? 2 : 1;
    for (int i = 0; i < 16; i++) {
        if (cell(b, i) == 0 && k-- == 0) return b | put(val, i);
    }
    return b;
}

static inline uint64_t game_seed(uint64_t seed, long g) {
    return seed * 0x9E3779B97F4A7C15ULL + (uint64_t)g * 0xD1B54A32D192ED03ULL + 0x632BE59BD9B4E019ULL;
}

/* --------------------------------------------------------------- network --- */

typedef struct {
    board_t key;
    float val;
    uint8_t depth;
} tt_entry_t;

typedef struct search_ctx {
    tt_entry_t *tt;
    size_t mask;
    double cutoff;
    struct net *net;
    uint64_t version;
} search_ctx_t;

/* Stages: a board's stage is the number of tile-mass boundaries it has passed;
 * each stage owns a table set. The first `tc_stages` stages learn with temporal
 * coherence (per-entry E/A sums, 8 extra bytes per weight); the rest with plain
 * TD and one "touched" bit per entry. Promotion: an entry a stage never touched
 * reads through to the previous stage. `flags` is 0 for weights-only nets, where
 * promotion has been baked into the tables. */
typedef struct net {
    int K;
    int tc_stages;
    int flags;
    int n_stages;            /* = n_bounds + 1 */
    int n_bounds;
    int64_t bounds[MAX_STAGES];
    int len[MAX_PATTERNS];
    int orig[MAX_PATTERNS][MAX_LEN];
    int cells[MAX_PATTERNS][8][MAX_LEN];
    size_t size[MAX_PATTERNS];
    float *w[MAX_STAGES][MAX_PATTERNS], *e[MAX_STAGES][MAX_PATTERNS], *a[MAX_STAGES][MAX_PATTERNS];
    uint8_t *bits[MAX_STAGES][MAX_PATTERNS];
    uint64_t version;        /* bumped whenever weights change; search contexts flush on mismatch */
    search_ctx_t *main_ctx;  /* for single-threaded best_move calls */
    search_ctx_t *choose_ctx; /* same, for the choose-mode search (its own transposition table) */
} net_t;

static inline int64_t board_mass(board_t b) {
    int64_t m = 0;
    for (int i = 0; i < 16; i++) {
        int v = (int)cell(b, i);
        if (v) m += (int64_t)1 << v;
    }
    return m;
}

static inline int stage_of(board_t b, const int64_t *bounds, int n_bounds) {
    if (n_bounds == 0) return 0;
    int64_t m = board_mass(b);
    int s = 0;
    while (s < n_bounds && m >= bounds[s]) s++;
    return s;
}

int nt_stage_of(uint64_t lo, uint64_t hi, const int64_t *bounds, int n_bounds) { return stage_of(mk(lo, hi), bounds, n_bounds); }

static int sym_cell(int cell, int s) {
    int r = cell / 4, c = cell % 4;
    if (s & 4) c = 3 - c;
    for (int i = 0; i < (s & 3); i++) {
        int nr = 3 - c, nc = r;
        r = nr;
        c = nc;
    }
    return r * 4 + c;
}

/* Snake-order score: the tiles read along the path 0 1 2 3 / 7 6 5 4 / 8 9 10 11 /
 * 15 14 13 12, each weighted by 0.5^k, best of the 8 symmetries. A chain laid out
 * as a snake scores highest; a scattered chain loses the weight of every tile
 * that is off its place. Added to the beam's leaf value with a tunable weight
 * so the search keeps the chain in a shape it can collapse (and that looks good). */
static const int SNAKE_PATH[16] = {0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12};
static int snake_cells[8][16];
static double snake_w[16];
static pthread_once_t snake_once = PTHREAD_ONCE_INIT;

static void snake_init(void) {
    for (int sym = 0; sym < 8; sym++)
        for (int k = 0; k < 16; k++) snake_cells[sym][k] = sym_cell(SNAKE_PATH[k], sym);
    for (int k = 0; k < 16; k++) snake_w[k] = pow(0.5, k);
}

static double snake_score(board_t b) {
    pthread_once(&snake_once, snake_init);
    double best = 0.0;
    for (int sym = 0; sym < 8; sym++) {
        double acc = 0.0;
        for (int k = 0; k < 16; k++) {
            int v = cell(b, snake_cells[sym][k]);
            if (v) acc += (double)(1u << v) * snake_w[k];
        }
        if (acc > best) best = acc;
    }
    return best;
}

double nt_snake_score(uint64_t lo, uint64_t hi) { return snake_score(mk(lo, hi)); }

static inline int stage_is_tc(const net_t *net, int st) { return st < net->tc_stages; }
static inline int stage_has_bits(const net_t *net, int st) { return net->flags && st > 0 && !stage_is_tc(net, st); }

static net_t *net_alloc(const int *cells_flat, const int *lens, int K, int tc_stages, int flags,
                        const int64_t *bounds, int n_bounds) {
    init_tables();
    if (K < 1 || K > MAX_PATTERNS || n_bounds < 0 || n_bounds >= MAX_STAGES) return NULL;
    net_t *net = (net_t *)calloc(1, sizeof(net_t));
    net->K = K;
    net->n_bounds = n_bounds;
    net->n_stages = n_bounds + 1;
    if (tc_stages < 0) tc_stages = 0;
    if (tc_stages > net->n_stages) tc_stages = net->n_stages;
    net->tc_stages = tc_stages;
    net->flags = flags && net->n_stages > 1;
    for (int i = 0; i < n_bounds; i++) net->bounds[i] = bounds[i];
    for (int k = 0; k < K; k++) {
        int n = lens[k];
        if (n < 1 || n > 7) { free(net); return NULL; }   /* 18^7 entries is the most that fits an index */
        net->len[k] = n;
        for (int j = 0; j < n; j++) net->orig[k][j] = cells_flat[k * MAX_LEN + j];
        for (int s = 0; s < 8; s++)
            for (int j = 0; j < n; j++) net->cells[k][s][j] = sym_cell(net->orig[k][j], s);
        net->size[k] = 1;
        for (int j = 0; j < n; j++) net->size[k] *= RADIX;
        for (int st = 0; st < net->n_stages; st++) {
            net->w[st][k] = (float *)calloc(net->size[k], sizeof(float));
            if (stage_is_tc(net, st)) {
                net->e[st][k] = (float *)calloc(net->size[k], sizeof(float));
                net->a[st][k] = (float *)calloc(net->size[k], sizeof(float));
            }
            if (stage_has_bits(net, st)) net->bits[st][k] = (uint8_t *)calloc(net->size[k] / 8, 1);
        }
    }
    net->version = 1;
    return net;
}

void *nt_create_ex(const int *cells_flat, const int *lens, int K, int tc_stages,
                   const int64_t *bounds, int n_bounds) {
    return net_alloc(cells_flat, lens, K, tc_stages, 1, bounds, n_bounds);
}

void *nt_create(const int *cells_flat, const int *lens, int K, int use_tc,
                const int64_t *bounds, int n_bounds) {
    return net_alloc(cells_flat, lens, K, use_tc ? n_bounds + 1 : 0, 1, bounds, n_bounds);
}

void nt_free(void *p) {
    net_t *net = (net_t *)p;
    if (!net) return;
    for (int st = 0; st < net->n_stages; st++)
        for (int k = 0; k < net->K; k++) {
            free(net->w[st][k]); free(net->e[st][k]); free(net->a[st][k]); free(net->bits[st][k]);
        }
    if (net->main_ctx) { free(net->main_ctx->tt); free(net->main_ctx); }
    if (net->choose_ctx) { free(net->choose_ctx->tt); free(net->choose_ctx); }
    free(net);
}

/* Pin every table in RAM so random-access lookups never hit swap. 0 on success. */
int nt_lock_memory(void *p) {
    net_t *net = (net_t *)p;
    int rc = 0;
    for (int st = 0; st < net->n_stages; st++) {
        for (int k = 0; k < net->K; k++) {
            size_t bytes = net->size[k] * sizeof(float);
            if (net->w[st][k] && mlock(net->w[st][k], bytes) != 0) rc = -1;
            if (net->e[st][k] && mlock(net->e[st][k], bytes) != 0) rc = -1;
            if (net->a[st][k] && mlock(net->a[st][k], bytes) != 0) rc = -1;
            if (net->bits[st][k] && mlock(net->bits[st][k], net->size[k] / 8) != 0) rc = -1;
        }
    }
    return rc;
}

int nt_num_patterns(void *p) { return ((net_t *)p)->K; }
int nt_use_tc(void *p) { return ((net_t *)p)->tc_stages > 0; }
int nt_tc_stages(void *p) { return ((net_t *)p)->tc_stages; }
int nt_num_bounds(void *p) { return ((net_t *)p)->n_bounds; }
void nt_bounds(void *p, int64_t *out) {
    net_t *net = (net_t *)p;
    for (int i = 0; i < net->n_bounds; i++) out[i] = net->bounds[i];
}
int nt_pattern(void *p, int k, int *out) {
    net_t *net = (net_t *)p;
    for (int j = 0; j < net->len[k]; j++) out[j] = net->orig[k][j];
    return net->len[k];
}

static inline uint32_t tuple_index(board_t b, const int *cells, int n) {
    uint32_t idx = 0;
    for (int j = 0; j < n; j++) idx = idx * RADIX + (uint32_t)cell(b, cells[j]);
    return idx;
}

static inline int touched(const net_t *net, int st, int k, uint32_t i) {
    if (!net->flags || st == 0) return 1;
    if (stage_is_tc(net, st)) return net->a[st][k][i] != 0.0f;
    return (net->bits[st][k][i >> 3] >> (i & 7)) & 1;
}

static inline int entry_source(const net_t *net, int st, int k, uint32_t i) {
    while (st > 0 && !touched(net, st, k, i)) st--;
    return st;
}

static inline float entry_value(const net_t *net, int st, int k, uint32_t i) {
    return net->w[entry_source(net, st, k, i)][k][i];
}

static inline float net_value_stage(const net_t *net, board_t b, int st) {
    float v = 0.0f;
    for (int k = 0; k < net->K; k++)
        for (int s = 0; s < 8; s++) v += entry_value(net, st, k, tuple_index(b, net->cells[k][s], net->len[k]));
    return v;
}

static inline float net_value(const net_t *net, board_t b) {
    return net_value_stage(net, b, stage_of(b, net->bounds, net->n_bounds));
}

static inline void net_update_stage(net_t *net, board_t b, float delta, float alpha, float alpha_plain, int st) {
    int tc = stage_is_tc(net, st);
    float step = (tc ? alpha : alpha_plain) * delta / (float)(net->K * 8);
    for (int k = 0; k < net->K; k++) {
        for (int s = 0; s < 8; s++) {
            uint32_t i = tuple_index(b, net->cells[k][s], net->len[k]);
            if (st > 0 && !touched(net, st, k, i)) {   /* promotion: start from the previous stage's entry */
                int src = entry_source(net, st - 1, k, i);
                net->w[st][k][i] = net->w[src][k][i];
                if (tc && stage_is_tc(net, src)) {      /* and inherit its adaptive-rate state */
                    net->e[st][k][i] = net->e[src][k][i];
                    net->a[st][k][i] = net->a[src][k][i];
                }
            }
            if (tc) {
                float E = net->e[st][k][i], A = net->a[st][k][i];
                float beta = (A == 0.0f) ? 1.0f : fabsf(E) / A;
                net->w[st][k][i] += step * beta;
                net->e[st][k][i] = E + delta;
                net->a[st][k][i] = A + fabsf(delta);
            } else {
                net->w[st][k][i] += step;
                if (stage_has_bits(net, st)) net->bits[st][k][i >> 3] |= (uint8_t)(1 << (i & 7));
            }
        }
    }
}

static inline void net_update(net_t *net, board_t b, float delta, float alpha, float alpha_plain) {
    net_update_stage(net, b, delta, alpha, alpha_plain, stage_of(b, net->bounds, net->n_bounds));
}

/* ------------------------------------------------------------------ public --- */

/* Boards cross the C API as two 64-bit halves. */
void nt_move(uint64_t lo, uint64_t hi, int dir, int *reward, uint64_t *out_lo, uint64_t *out_hi) {
    init_tables();
    board_t r = do_move(mk(lo, hi), dir, reward);
    *out_lo = lo64(r);
    *out_hi = hi64(r);
}

void nt_spawn(uint64_t lo, uint64_t hi, uint64_t *rng, uint64_t *out_lo, uint64_t *out_hi) {
    board_t r = spawn(mk(lo, hi), rng);
    *out_lo = lo64(r);
    *out_hi = hi64(r);
}

int nt_cell_bits(void) { return CB; }
int nt_max_exp(void) { return MAX_EXP; }

float nt_value(void *p, uint64_t lo, uint64_t hi) { return net_value((net_t *)p, mk(lo, hi)); }

void nt_update(void *p, uint64_t lo, uint64_t hi, float delta, float alpha, float alpha_plain) {
    net_t *net = (net_t *)p;
    net_update(net, mk(lo, hi), delta, alpha, alpha_plain);
    net->version++;
}

float nt_value_stage(void *p, uint64_t lo, uint64_t hi, int stage) {
    net_t *net = (net_t *)p;
    board_t b = mk(lo, hi);
    if (stage < 0 || stage >= net->n_stages) return net_value(net, b);
    return net_value_stage(net, b, stage);
}

void nt_update_stage(void *p, uint64_t lo, uint64_t hi, float delta, float alpha, float alpha_plain, int stage) {
    net_t *net = (net_t *)p;
    board_t b = mk(lo, hi);
    if (stage < 0 || stage >= net->n_stages) net_update(net, b, delta, alpha, alpha_plain);
    else net_update_stage(net, b, delta, alpha, alpha_plain, stage);
    net->version++;
}

/* ---------------------------------------------------------------- learning --- */

static void learn_game(net_t *net, float alpha, float alpha_plain, uint64_t *rng, int64_t *score, int32_t *maxtile, int32_t *nmoves) {
    board_t b = spawn(spawn(0, rng), rng);
    board_t prev = 0;
    int have_prev = 0;
    int64_t sc = 0;
    int32_t mv = 0;
    for (;;) {
        int best = -1, bestr = 0;
        float bestv = 0.0f, bestval = 0.0f;
        board_t besta = 0;
        for (int d = 0; d < 4; d++) {
            int r;
            board_t a = do_move(b, d, &r);
            if (a == b) continue;
            float va = net_value(net, a);
            float v = (float)r + va;
            if (best < 0 || v > bestv) { bestv = v; best = d; besta = a; bestr = r; bestval = va; }
        }
        if (best < 0) {
            if (have_prev) net_update(net, prev, 0.0f - net_value(net, prev), alpha, alpha_plain);
            break;
        }
        if (have_prev) net_update(net, prev, (float)bestr + bestval - net_value(net, prev), alpha, alpha_plain);
        prev = besta;
        have_prev = 1;
        sc += bestr;
        mv++;
        b = spawn(besta, rng);
    }
    *score = sc;
    *maxtile = max_exp(b) ? (1 << max_exp(b)) : 0;
    *nmoves = mv;
}

typedef struct {
    net_t *net;
    long *next, n;   /* shared atomic counter: fast cores simply take more games */
    float alpha, alpha_plain;
    uint64_t seed;
    int64_t *scores;
    int32_t *maxtiles, *moves;
} train_job_t;

static void *train_worker(void *arg) {
    train_job_t *j = (train_job_t *)arg;
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        learn_game(j->net, j->alpha, j->alpha_plain, &rng, &j->scores[g], &j->maxtiles[g], &j->moves[g]);
    }
    return NULL;
}

void nt_train(void *p, long games, int threads, float alpha, float alpha_plain, uint64_t seed,
              int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    net_t *net = (net_t *)p;
    if (threads < 1) threads = 1;
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * threads);
    train_job_t *jobs = (train_job_t *)malloc(sizeof(train_job_t) * threads);
    long next = 0;
    for (int t = 0; t < threads; t++) {
        jobs[t] = (train_job_t){net, &next, games, alpha, alpha_plain, seed, scores, maxtiles, moves};
        pthread_create(&tid[t], NULL, train_worker, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    free(tid);
    free(jobs);
    net->version++;
}

/* ------------------------------------------------------------- expectimax --- */

/* use_tt = 0 gives a context without a transposition table, for searches made
 * while the weights are being updated (cached values would go stale at once). */
static search_ctx_t *ctx_create_tt(net_t *net, double cutoff, int use_tt) {
    search_ctx_t *c = (search_ctx_t *)calloc(1, sizeof(search_ctx_t));
    c->tt = use_tt ? (tt_entry_t *)calloc((size_t)1 << TT_BITS, sizeof(tt_entry_t)) : NULL;
    c->mask = ((size_t)1 << TT_BITS) - 1;
    c->cutoff = cutoff;
    c->net = net;
    c->version = net->version;
    return c;
}

static search_ctx_t *ctx_create(net_t *net, double cutoff) {
    return ctx_create_tt(net, cutoff, 1);
}

static void ctx_sync(search_ctx_t *c, double cutoff) {
    if (c->version != c->net->version || c->cutoff != cutoff) {
        if (c->tt) memset(c->tt, 0, ((size_t)1 << TT_BITS) * sizeof(tt_entry_t));
        c->version = c->net->version;
        c->cutoff = cutoff;
    }
}

static void ctx_free(search_ctx_t *c) {
    if (!c) return;
    free(c->tt);
    free(c);
}

static double max_node(search_ctx_t *c, board_t b, int depth, double prob);

static double chance_node(search_ctx_t *c, board_t after, int depth, double prob) {
    if (depth == 0 || prob < c->cutoff) return net_value(c->net, after);
    size_t h = tt_hash(after);
    tt_entry_t *e = &c->tt[h];
    if (e->key == after && e->depth == depth) return e->val;
    int empty = count_empty(after);
    double p2 = prob * 0.9 / empty, p4 = prob * 0.1 / empty, sum = 0.0;
    for (int i = 0; i < 16; i++) {
        if (cell(after, i) != 0) continue;
        sum += 0.9 * max_node(c, after | (put(1, i)), depth - 1, p2);
        sum += 0.1 * max_node(c, after | (put(2, i)), depth - 1, p4);
    }
    double v = sum / empty;
    e->key = after;
    e->depth = (uint8_t)depth;
    e->val = (float)v;
    return v;
}

static double max_node(search_ctx_t *c, board_t b, int depth, double prob) {
    double best = 0.0;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        int r;
        board_t a = do_move(b, d, &r);
        if (a == b) continue;
        double v = (double)r + chance_node(c, a, depth, prob);
        if (!any || v > best) { best = v; any = 1; }
    }
    return best;
}

static int best_move(search_ctx_t *c, board_t b, int depth) {
    int best = -1;
    double bestv = 0.0;
    for (int d = 0; d < 4; d++) {
        int r;
        board_t a = do_move(b, d, &r);
        if (a == b) continue;
        double v = (double)r + chance_node(c, a, depth, 1.0);
        if (best < 0 || v > bestv) { bestv = v; best = d; }
    }
    return best;
}

static search_ctx_t *main_ctx(net_t *net, double cutoff) {
    if (!net->main_ctx) net->main_ctx = ctx_create(net, cutoff);
    ctx_sync(net->main_ctx, cutoff);
    return net->main_ctx;
}

int nt_best_move(void *p, uint64_t lo, uint64_t hi, int depth, double cutoff) {
    net_t *net = (net_t *)p;
    return best_move(main_ctx(net, cutoff), mk(lo, hi), depth);
}

double nt_search_value(void *p, uint64_t lo, uint64_t hi, int depth, double cutoff) {
    net_t *net = (net_t *)p;
    return max_node(main_ctx(net, cutoff), mk(lo, hi), depth, 1.0);
}

typedef struct {
    net_t *net;
    int depth;
    double cutoff;
    long *next, n;
    uint64_t seed;
    int64_t *scores;
    int32_t *maxtiles, *moves;
} play_job_t;

static void *play_worker(void *arg) {
    play_job_t *j = (play_job_t *)arg;
    search_ctx_t *c = ctx_create(j->net, j->cutoff);
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        board_t b = spawn(spawn(0, &rng), &rng);
        int64_t sc = 0;
        int32_t mv = 0;
        for (;;) {
            int d = best_move(c, b, j->depth);
            if (d < 0) break;
            int r;
            b = spawn(do_move(b, d, &r), &rng);
            sc += r;
            mv++;
        }
        j->scores[g] = sc;
        j->maxtiles[g] = max_exp(b) ? (1 << max_exp(b)) : 0;
        j->moves[g] = mv;
    }
    ctx_free(c);
    return NULL;
}

void nt_play(void *p, long games, int depth, double cutoff, int threads, uint64_t seed,
             int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    net_t *net = (net_t *)p;
    if (threads < 1) threads = 1;
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * threads);
    play_job_t *jobs = (play_job_t *)malloc(sizeof(play_job_t) * threads);
    long next = 0;
    for (int t = 0; t < threads; t++) {
        jobs[t] = (play_job_t){net, depth, cutoff, &next, games, seed, scores, maxtiles, moves};
        pthread_create(&tid[t], NULL, play_worker, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    free(tid);
    free(jobs);
}

/* A random spawn that never ends the game when it can be avoided: uniform over
 * the (cell, tile) placements after which a move exists, else any. Used for the
 * random prefix of evaluation games: the choose policy picks moves assuming it
 * will place the tile, and plain random spawns kill it within a few hundred
 * moves; kind spawns keep the games alive yet different. */
static board_t spawn_kind(board_t b, uint64_t *rng) {
    board_t ok[32], any[32];
    int n_ok = 0, n_any = 0;
    for (int i = 0; i < 16; i++) {
        if (cell(b, i)) continue;
        for (board_t v = 1; v <= 2; v++) {
            board_t s = b | put((int)v, i);
            any[n_any++] = s;
            int alive = 0;
            for (int m = 0; m < 4 && !alive; m++) { int r; alive = do_move(s, m, &r) != s; }
            if (alive) ok[n_ok++] = s;
        }
    }
    if (n_any == 0) return b;
    uint64_t x = rng_next(rng);
    return n_ok ? ok[x % (uint64_t)n_ok] : any[x % (uint64_t)n_any];
}

/* ---------------------------------------------------- choose-mode search --- */
/* The player places every tile, so the game is a pure maximisation:
 *   MV(s, d) = max over moves of reward + AV(after, d)      (dead board: 0)
 *   AV(a, 0) = V(a)
 *   AV(a, d) = max over placements of MV(placed, d - 1), recursing only into the
 *              top-k placements ranked by MV(placed, 0)
 * depth counts placement decisions searched. */

typedef struct { board_t s; double q; } cand_t;

static double choose_mv(search_ctx_t *c, board_t s, int depth, int topk);

/* Value of the best placement on afterstate `a`; *best_s receives that board. */
static double choose_place(search_ctx_t *c, board_t a, int depth, int topk, board_t *best_s) {
    cand_t cands[32];
    int n = 0;
    for (int i = 0; i < 16; i++) {
        if (cell(a, i)) continue;
        for (board_t v = 1; v <= 2; v++) {
            cands[n].s = a | put((int)v, i);
            cands[n].q = choose_mv(c, cands[n].s, 0, topk);
            n++;
        }
    }
    if (n == 0) { *best_s = a; return 0.0; }
    int k = depth <= 1 ? 1 : (topk < n ? topk : n);
    for (int i = 0; i < k; i++) {           /* partial selection sort: best k first */
        int m = i;
        for (int j = i + 1; j < n; j++) if (cands[j].q > cands[m].q) m = j;
        cand_t t = cands[i]; cands[i] = cands[m]; cands[m] = t;
    }
    if (depth <= 1) { *best_s = cands[0].s; return cands[0].q; }
    double best = -1e300;
    for (int i = 0; i < k; i++) {
        double v = choose_mv(c, cands[i].s, depth - 1, topk);
        if (v > best) { best = v; *best_s = cands[i].s; }
    }
    return best;
}

static double choose_av(search_ctx_t *c, board_t a, int depth, int topk) {
    if (depth == 0) return net_value(c->net, a);
    tt_entry_t *e = NULL;
    if (c->tt) {
        size_t h = tt_hash(a);
        e = &c->tt[h];
        if (e->key == a && e->depth == depth) return e->val;
    }
    board_t dummy;
    double v = choose_place(c, a, depth, topk, &dummy);
    if (e) {
        e->key = a;
        e->depth = (uint8_t)depth;
        e->val = (float)v;
    }
    return v;
}

static double choose_mv(search_ctx_t *c, board_t s, int depth, int topk) {
    double best = 0.0;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        int r;
        board_t a = do_move(s, d, &r);
        if (a == s) continue;
        double v = (double)r + choose_av(c, a, depth, topk);
        if (!any || v > best) { best = v; any = 1; }
    }
    return best;
}

static int best_choose(search_ctx_t *c, board_t b, int depth, int topk, int *cell_out, int *value) {
    int best_m = -1;
    double best_v = 0.0;
    board_t best_a = 0;
    for (int d = 0; d < 4; d++) {
        int r;
        board_t a = do_move(b, d, &r);
        if (a == b) continue;
        double v = (double)r + choose_av(c, a, depth, topk);
        if (best_m < 0 || v > best_v) { best_v = v; best_m = d; best_a = a; }
    }
    if (best_m < 0) return -1;
    board_t placed;
    choose_place(c, best_a, depth, topk, &placed);
    board_t diff = placed ^ best_a;
    int idx = 0;
    while (idx < 16 && cell(diff, idx) == 0) idx++;
    *cell_out = idx < 16 ? idx : -1;
    *value = idx < 16 ? cell(placed, idx) : 0;
    return best_m;
}

static search_ctx_t *choose_ctx(net_t *net) {
    if (!net->choose_ctx) net->choose_ctx = ctx_create(net, 0.0);
    ctx_sync(net->choose_ctx, 0.0);
    return net->choose_ctx;
}

int nt_best_choose(void *p, uint64_t lo, uint64_t hi, int depth, int topk, int *cell_out, int *value) {
    net_t *net = (net_t *)p;
    if (depth < 1) depth = 1;
    if (topk < 1) topk = 1;
    return best_choose(choose_ctx(net), mk(lo, hi), depth, topk, cell_out, value);
}

double nt_choose_value(void *p, uint64_t lo, uint64_t hi, int depth, int topk) {
    net_t *net = (net_t *)p;
    if (topk < 1) topk = 1;
    return choose_mv(choose_ctx(net), mk(lo, hi), depth < 0 ? 0 : depth, topk);
}

typedef struct {
    net_t *net;
    int depth, topk;
    long *next, n, max_moves, prefix;   /* prefix: moves with kind random spawns first, for varied games */
    uint64_t seed;
    int64_t *scores;
    int32_t *maxtiles, *moves;
} choose_job_t;

static void *choose_worker(void *arg) {
    choose_job_t *j = (choose_job_t *)arg;
    search_ctx_t *c = ctx_create(j->net, 0.0);
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        board_t b = spawn(spawn(0, &rng), &rng);
        int64_t sc = 0;
        int32_t mv = 0;
        while (mv < j->max_moves) {
            int cell, value;
            int d = best_choose(c, b, j->depth, j->topk, &cell, &value);
            if (d < 0) break;
            int r;
            b = do_move(b, d, &r);
            if (mv < j->prefix) b = spawn_kind(b, &rng);
            else if (cell >= 0) b |= put(value, cell);
            sc += r;
            mv++;
        }
        j->scores[g] = sc;
        j->maxtiles[g] = max_exp(b) ? (1 << max_exp(b)) : 0;
        j->moves[g] = mv;
    }
    ctx_free(c);
    return NULL;
}

void nt_play_choose(void *p, long games, int depth, int topk, int threads, uint64_t seed, long max_moves,
                    long prefix, int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    net_t *net = (net_t *)p;
    if (threads < 1) threads = 1;
    if (depth < 1) depth = 1;
    if (topk < 1) topk = 1;
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * threads);
    choose_job_t *jobs = (choose_job_t *)malloc(sizeof(choose_job_t) * threads);
    long next = 0;
    for (int t = 0; t < threads; t++) {
        jobs[t] = (choose_job_t){net, depth, topk, &next, games, max_moves, prefix, seed, scores, maxtiles, moves};
        pthread_create(&tid[t], NULL, choose_worker, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    free(tid);
    free(jobs);
}

/* ------------------------------------------------------------ beam search --- */
/* The choose game is deterministic, so a line of play can be planned like a
 * puzzle: keep the `width` best boards, extend each by a move and a placement,
 * repeat `depth` times. An entry is scored by the rewards along its line plus
 * the best next move's reward + V(afterstate) (a dead board adds 0 and is
 * carried along unexpanded). With unlimited width this equals the full-width
 * max-max tree of the same depth. A board reached twice is kept once. Levels
 * are kept so the best line can be read back through parent links. `spread` > 0
 * caps how many survivors of a level may share a parent (from level 2 on, since
 * level 1 has only the root), so the beam covers distinct lines instead of
 * near-duplicates of the best one. */

typedef struct {
    board_t s;               /* board awaiting a move (after the placement) */
    double score;            /* cum + best next-move value */
    int64_t cum;             /* rewards collected along the line */
    int32_t parent;          /* index in the previous level */
    int8_t alive;            /* 0 when s has no legal move */
    int8_t m, cell, val;     /* the step that produced s; m = -1 for a carried dead entry */
} beam_t;

static inline int next_value(net_t *net, board_t s, double snake, double *out) {
    double best = 0.0;
    int any = 0;
    for (int m = 0; m < 4; m++) {
        int r;
        board_t a = do_move(s, m, &r);
        if (a == s) continue;
        double v = (double)r + net_value(net, a);
        if (snake > 0.0) v += snake * snake_score(a);
        if (!any || v > best) { best = v; any = 1; }
    }
    *out = best;
    return any;
}

static int beam_cmp(const void *x, const void *y) {
    const beam_t *a = (const beam_t *)x, *b = (const beam_t *)y;
    if (a->score != b->score) return a->score > b->score ? -1 : 1;
    return a->s < b->s ? -1 : (a->s > b->s ? 1 : 0);   /* boards are unique: a total order */
}

/* Writes the best line's steps to moves/cells/values (each of size `depth`) and
 * returns its length (0 if the root has no move); *score gets the line's value. */
static int beam_search(net_t *net, board_t b, int width, int depth, int spread, double snake, int *moves,
                       int *cells, int *values, double *score) {
    if (width < 1) width = 1;
    if (depth < 1) depth = 1;
    if (spread < 0) spread = 0;
    size_t cap = (size_t)width * 121;            /* 4 moves x 30 placements, plus carried dead lines */
    size_t hcap = 1;
    while (hcap < 2 * cap) hcap <<= 1;
    int hbits = 0;
    while (((size_t)1 << hbits) < hcap) hbits++;
    beam_t *lev = (beam_t *)malloc(sizeof(beam_t) * (size_t)width * (size_t)(depth + 1));
    beam_t *cand = (beam_t *)malloc(sizeof(beam_t) * cap);
    board_t *seen = (board_t *)malloc(sizeof(board_t) * hcap);
    int *kids = (int *)malloc(sizeof(int) * (size_t)width);
    lev[0] = (beam_t){b, 0.0, 0, -1, 1, -1, -1, 0};
    int n_cur = 1, L = 0;
    for (int d = 0; d < depth; d++) {
        beam_t *cur = lev + (size_t)d * width;
        size_t n_cand = 0;
        memset(seen, 0, sizeof(board_t) * hcap);
        for (int i = 0; i < n_cur; i++) {
            const beam_t *e = &cur[i];
            if (!e->alive) {                     /* a finished line still competes */
                if (d > 0) {
                    beam_t ne = *e;
                    ne.parent = i;
                    ne.m = -1;
                    cand[n_cand++] = ne;
                }
                continue;
            }
            for (int m = 0; m < 4; m++) {
                int r;
                board_t a = do_move(e->s, m, &r);
                if (a == e->s) continue;
                for (int c = 0; c < 16; c++) {
                    if (cell(a, c)) continue;
                    for (board_t v = 1; v <= 2; v++) {
                        board_t s2 = a | put((int)v, c);
                        size_t h = (size_t)(hash128(s2) >> (64 - hbits));
                        while (seen[h] && seen[h] != s2) h = (h + 1) & (hcap - 1);
                        if (seen[h] == s2) continue;
                        seen[h] = s2;
                        beam_t ne;
                        ne.s = s2;
                        ne.cum = e->cum + r;
                        ne.parent = i;
                        double nv;
                        ne.alive = (int8_t)next_value(net, s2, snake, &nv);
                        ne.score = (double)ne.cum + (ne.alive ? nv : 0.0);
                        ne.m = (int8_t)m;
                        ne.cell = (int8_t)c;
                        ne.val = (int8_t)v;
                        cand[n_cand++] = ne;
                    }
                }
            }
        }
        if (n_cand == 0) break;                  /* root had no move */
        qsort(cand, n_cand, sizeof(beam_t), beam_cmp);
        beam_t *nxt = lev + (size_t)(d + 1) * width;
        n_cur = 0;
        if (spread > 0 && d > 0) {
            memset(kids, 0, sizeof(int) * (size_t)width);
            for (size_t i = 0; i < n_cand && n_cur < width; i++) {
                if (kids[cand[i].parent] >= spread) continue;
                kids[cand[i].parent]++;
                nxt[n_cur++] = cand[i];
            }
        } else {
            n_cur = (int)(n_cand < (size_t)width ? n_cand : (size_t)width);
            memcpy(nxt, cand, sizeof(beam_t) * (size_t)n_cur);
        }
        L = d + 1;
        int any_alive = 0;
        for (int i = 0; i < n_cur; i++) any_alive |= lev[(size_t)L * width + i].alive;
        if (!any_alive) break;
    }
    int len = 0;
    if (L > 0) {                                 /* read the line back through the parents */
        int idx = 0;
        for (int d = L; d > 0; d--) {
            const beam_t *e = &lev[(size_t)d * width + idx];
            if (e->m >= 0) len++;
            idx = e->parent;
        }
        int k = len, idx2 = 0;
        for (int d = L; d > 0; d--) {
            const beam_t *e = &lev[(size_t)d * width + idx2];
            if (e->m >= 0) { k--; moves[k] = e->m; cells[k] = e->cell; values[k] = e->val; }
            idx2 = e->parent;
        }
        *score = lev[(size_t)L * width].score;
    } else {
        *score = 0.0;
    }
    free(lev);
    free(cand);
    free(seen);
    free(kids);
    return len;
}

#define BEAM_MAX_DEPTH 256

int nt_beam_plan(void *p, uint64_t lo, uint64_t hi, int width, int depth, int spread, double snake,
                 int *moves, int *cells, int *values) {
    board_t b = mk(lo, hi);
    double score;
    if (depth > BEAM_MAX_DEPTH) depth = BEAM_MAX_DEPTH;
    return beam_search((net_t *)p, b, width, depth, spread, snake, moves, cells, values, &score);
}

int nt_beam_choose(void *p, uint64_t lo, uint64_t hi, int width, int depth, int spread, double snake,
                   int *cell_out, int *value) {
    board_t b = mk(lo, hi);
    int moves[BEAM_MAX_DEPTH], cells[BEAM_MAX_DEPTH], values[BEAM_MAX_DEPTH];
    double score;
    if (depth > BEAM_MAX_DEPTH) depth = BEAM_MAX_DEPTH;
    int len = beam_search((net_t *)p, b, width, depth, spread, snake, moves, cells, values, &score);
    if (len == 0) { *cell_out = -1; *value = 0; return -1; }
    *cell_out = cells[0];
    *value = values[0];
    return moves[0];
}

double nt_beam_value(void *p, uint64_t lo, uint64_t hi, int width, int depth, int spread, double snake) {
    board_t b = mk(lo, hi);
    int moves[BEAM_MAX_DEPTH], cells[BEAM_MAX_DEPTH], values[BEAM_MAX_DEPTH];
    double score;
    if (depth > BEAM_MAX_DEPTH) depth = BEAM_MAX_DEPTH;
    beam_search((net_t *)p, b, width, depth, spread, snake, moves, cells, values, &score);
    return score;
}

typedef struct {
    net_t *net;
    int width, depth, stride, spread;
    double snake;
    long *next, n, max_moves, prefix;
    uint64_t seed;
    int64_t *scores;
    int32_t *maxtiles, *moves;
} beam_job_t;

static void *beam_worker(void *arg) {
    beam_job_t *j = (beam_job_t *)arg;
    int moves[BEAM_MAX_DEPTH], cells[BEAM_MAX_DEPTH], values[BEAM_MAX_DEPTH];
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        board_t b = spawn(spawn(0, &rng), &rng);
        int64_t sc = 0;
        int32_t mv = 0;
        while (mv < j->max_moves) {
            double score;
            int len = beam_search(j->net, b, j->width, j->depth, j->spread, j->snake, moves, cells, values, &score);
            if (len == 0) break;
            int take = j->stride < len ? j->stride : len;   /* commit the first `stride` steps */
            if (mv < j->prefix) take = 1;                   /* a random spawn breaks the plan */
            for (int k = 0; k < take && mv < j->max_moves; k++) {
                int r;
                b = do_move(b, moves[k], &r);
                if (mv < j->prefix) b = spawn_kind(b, &rng);
                else b |= put(values[k], cells[k]);
                sc += r;
                mv++;
            }
        }
        j->scores[g] = sc;
        j->maxtiles[g] = max_exp(b) ? (1 << max_exp(b)) : 0;
        j->moves[g] = mv;
    }
    return NULL;
}

void nt_play_beam(void *p, long games, int width, int depth, int stride, int spread, double snake, int threads,
                  uint64_t seed, long max_moves, long prefix, int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    net_t *net = (net_t *)p;
    if (threads < 1) threads = 1;
    if (stride < 1) stride = 1;
    if (depth > BEAM_MAX_DEPTH) depth = BEAM_MAX_DEPTH;
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * threads);
    beam_job_t *jobs = (beam_job_t *)malloc(sizeof(beam_job_t) * threads);
    long next = 0;
    for (int t = 0; t < threads; t++) {
        jobs[t] = (beam_job_t){net, width, depth, stride, spread, snake, &next, games, max_moves, prefix, seed, scores, maxtiles, moves};
        pthread_create(&tid[t], NULL, beam_worker, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    free(tid);
    free(jobs);
}

/* ------------------------------------------------ learning the choose game --- */
/* TD(0) on afterstates, mirroring learn_game: the policy is the choose search at
 * `depth` (1 = every placement, then every reply move) using the current weights,
 * so the value function learns the game where the player places the tiles. */

static void learn_choose_game(net_t *net, search_ctx_t *c, int depth, int topk, long max_moves, double explore,
                              float alpha, float alpha_plain, uint64_t *rng,
                              int64_t *score, int32_t *maxtile, int32_t *nmoves) {
    board_t b = spawn(spawn(0, rng), rng);
    board_t prev = 0;
    int have_prev = 0;
    int64_t sc = 0;
    int32_t mv = 0;
    while (mv < max_moves) {
        int cell, value;
        int d = best_choose(c, b, depth, topk, &cell, &value);
        if (d < 0) {
            if (have_prev) net_update(net, prev, 0.0f - net_value(net, prev), alpha, alpha_plain);
            break;
        }
        int r;
        board_t a = do_move(b, d, &r);
        if (have_prev) net_update(net, prev, (float)r + net_value(net, a) - net_value(net, prev), alpha, alpha_plain);
        prev = a;
        have_prev = 1;
        sc += r;
        mv++;
        /* exploration: now and then place a uniformly random tile instead of the
         * searched one, so training games leave the single line a deterministic
         * policy would otherwise replay (on-policy TD: the noisy policy is the one
         * being evaluated, which also makes the values a little more robust) */
        if (explore > 0.0 && (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0) < explore) {
            b = spawn(a, rng);
        } else {
            b = a;
            if (cell >= 0) b |= put(value, cell);
        }
    }
    /* stopping at max_moves is not a terminal state, so no final update there */
    *score = sc;
    *maxtile = max_exp(b) ? (1 << max_exp(b)) : 0;
    *nmoves = mv;
}

typedef struct {
    net_t *net;
    int depth, topk;
    long *next, n, max_moves;
    double explore;
    float alpha, alpha_plain;
    uint64_t seed;
    int64_t *scores;
    int32_t *maxtiles, *moves;
} choose_train_job_t;

static void *choose_train_worker(void *arg) {
    choose_train_job_t *j = (choose_train_job_t *)arg;
    search_ctx_t *c = ctx_create_tt(j->net, 0.0, 0);
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        learn_choose_game(j->net, c, j->depth, j->topk, j->max_moves, j->explore, j->alpha, j->alpha_plain, &rng,
                          &j->scores[g], &j->maxtiles[g], &j->moves[g]);
    }
    ctx_free(c);
    return NULL;
}

void nt_train_choose(void *p, long games, int threads, float alpha, float alpha_plain, uint64_t seed,
                     int depth, int topk, long max_moves, double explore,
                     int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    net_t *net = (net_t *)p;
    if (threads < 1) threads = 1;
    if (depth < 1) depth = 1;
    if (topk < 1) topk = 1;
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * threads);
    choose_train_job_t *jobs = (choose_train_job_t *)malloc(sizeof(choose_train_job_t) * threads);
    long next = 0;
    for (int t = 0; t < threads; t++) {
        jobs[t] = (choose_train_job_t){net, depth, topk, &next, games, max_moves, explore, alpha, alpha_plain, seed,
                                       scores, maxtiles, moves};
        pthread_create(&tid[t], NULL, choose_train_worker, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    free(tid);
    free(jobs);
    net->version++;
}

/* ------------------------------------------------------------- persistence --- */

static const char MAGIC_V1[8] = "NTUPLE01";
static const char MAGIC_V2[8] = "NTUPLE02";
static const char MAGIC_V3[8] = "NTUPLE03";
static const char MAGIC_V4[8] = "NTUPLE04";
static const char MAGIC_V5[8] = "NTUPLE05";
static const char MAGIC_V6[8] = "NTUPLE06";   /* radix-18 tables (five-bit cells) */

/* Format 05: header (K, tc_stages, flags, n_bounds, bounds, len, orig), then per
 * stage and pattern: w; e and a for TC stages; touched bits for plain stages >= 1.
 * weights_only drops the learning state and writes every stage's table with
 * promotion applied, so the file plays identically at a third of the size. */
int nt_save_ex(void *p, const char *path, int weights_only) {
    net_t *net = (net_t *)p;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int tc_stages = weights_only ? 0 : net->tc_stages;
    int flags = weights_only ? 0 : net->flags;
    fwrite(MAGIC_V6, 1, 8, f);
    fwrite(&net->K, sizeof(int), 1, f);
    fwrite(&tc_stages, sizeof(int), 1, f);
    fwrite(&flags, sizeof(int), 1, f);
    fwrite(&net->n_bounds, sizeof(int), 1, f);
    fwrite(net->bounds, sizeof(int64_t), MAX_STAGES, f);
    fwrite(net->len, sizeof(int), MAX_PATTERNS, f);
    fwrite(net->orig, sizeof(int), MAX_PATTERNS * MAX_LEN, f);
    float *buf = NULL;
    for (int st = 0; st < net->n_stages; st++) {
        for (int k = 0; k < net->K; k++) {
            if (weights_only && st > 0 && net->flags) {
                if (!buf) buf = (float *)malloc(net->size[k] * sizeof(float));
                for (uint32_t i = 0; i < net->size[k]; i++) buf[i] = entry_value(net, st, k, i);
                fwrite(buf, sizeof(float), net->size[k], f);
            } else {
                fwrite(net->w[st][k], sizeof(float), net->size[k], f);
            }
            if (!weights_only && stage_is_tc(net, st)) {
                fwrite(net->e[st][k], sizeof(float), net->size[k], f);
                fwrite(net->a[st][k], sizeof(float), net->size[k], f);
            }
            if (!weights_only && stage_has_bits(net, st)) fwrite(net->bits[st][k], 1, net->size[k] / 8, f);
        }
    }
    free(buf);
    return fclose(f) == 0 ? 0 : -1;
}

int nt_save(void *p, const char *path) { return nt_save_ex(p, path, 0); }

static int skip_bytes(FILE *f, size_t n) { return fseek(f, (long)n, SEEK_CUR) == 0 ? 0 : -1; }

/* Formats 1-5 indexed tables with 4 bits per cell; entry `old` of an n-cell table
 * moves to this radix-18 index (entries with a digit above 15 stay zero). */
static inline uint32_t remap_old_index(uint32_t old, int n) {
    uint32_t idx = 0;
    for (int j = 0; j < n; j++) idx = idx * RADIX + ((old >> (4 * (n - 1 - j))) & 0xF);
    return idx;
}

/* Read a table of `fn` floats into dst (NULL = skip); re-index when the file is old. */
static int read_table(FILE *f, float *dst, size_t fn, int old, int n, float *tmp) {
    if (!dst) return skip_bytes(f, fn * sizeof(float));
    if (!old) return fread(dst, sizeof(float), fn, f) == fn ? 0 : -1;
    if (fread(tmp, sizeof(float), fn, f) != fn) return -1;
    for (uint32_t i = 0; i < fn; i++) dst[remap_old_index(i, n)] = tmp[i];
    return 0;
}

static int read_bits(FILE *f, uint8_t *dst, size_t fn, int old, int n, uint8_t *tmpb) {
    if (!dst) return skip_bytes(f, fn / 8);
    if (!old) return fread(dst, 1, fn / 8, f) == fn / 8 ? 0 : -1;
    if (fread(tmpb, 1, fn / 8, f) != fn / 8) return -1;
    for (uint32_t i = 0; i < fn; i++)
        if ((tmpb[i >> 3] >> (i & 7)) & 1) { uint32_t j = remap_old_index(i, n); dst[j >> 3] |= (uint8_t)(1 << (j & 7)); }
    return 0;
}

/* req_bounds/n_req: when given and the file is single-stage, its tables become
 * stage 0 of a network with those boundaries (later stages read through until
 * trained). A staged file must be loaded with no or identical boundaries.
 * tc_req: -1 keeps the file's learning mode; otherwise the number of leading
 * stages that keep temporal-coherence state (0 = plain TD everywhere, a third
 * of the memory). Reads every earlier format. */
void *nt_load_ex(const char *path, const int64_t *req_bounds, int n_req, int tc_req) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[8];
    int K, len[MAX_PATTERNS], orig[MAX_PATTERNS][MAX_LEN];
    int n_bounds = 0, f_tc_stages = 0, f_flags = 0;
    int f_e_all = 0, f_a_all = 0, f_bits_v4 = 0;   /* formats 1-4: per-file (not per-stage) layout */
    int64_t bounds[MAX_STAGES] = {0};
    if (fread(magic, 1, 8, f) != 8) { fclose(f); return NULL; }
    int v = memcmp(magic, MAGIC_V6, 8) == 0 ? 6 : memcmp(magic, MAGIC_V5, 8) == 0 ? 5 : memcmp(magic, MAGIC_V4, 8) == 0 ? 4 :
            memcmp(magic, MAGIC_V3, 8) == 0 ? 3 : memcmp(magic, MAGIC_V2, 8) == 0 ? 2 :
            memcmp(magic, MAGIC_V1, 8) == 0 ? 1 : 0;
    if (!v) { fclose(f); return NULL; }
    if (fread(&K, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (v >= 5) {
        if (fread(&f_tc_stages, sizeof(int), 1, f) != 1 || fread(&f_flags, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    } else {
        int use_tc, has_a = -1;
        if (fread(&use_tc, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
        if (v >= 3 && fread(&has_a, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
        if (v == 4 && fread(&f_bits_v4, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
        f_e_all = use_tc;
        f_a_all = has_a < 0 ? 1 : has_a;          /* formats 1-2 always stored A when it existed */
        if (has_a < 0) f_a_all = use_tc;
    }
    if (v >= 2 && (fread(&n_bounds, sizeof(int), 1, f) != 1 ||
                   fread(bounds, sizeof(int64_t), MAX_STAGES, f) != MAX_STAGES)) { fclose(f); return NULL; }
    if (fread(len, sizeof(int), MAX_PATTERNS, f) != MAX_PATTERNS ||
        fread(orig, sizeof(int), MAX_PATTERNS * MAX_LEN, f) != MAX_PATTERNS * MAX_LEN) { fclose(f); return NULL; }
    int file_stages = n_bounds + 1;
    if (v < 5) {
        if (v <= 2 && !f_e_all && n_bounds > 0) f_a_all = 1;   /* old plain staged nets used A as flags */
        f_tc_stages = f_e_all ? file_stages : 0;
        f_flags = (f_a_all && n_bounds > 0) || f_bits_v4;
    }
    const int64_t *target = bounds;
    int n_target = n_bounds;
    int restaged = 0;
    if (n_req > 0) {
        if (n_bounds == 0) { target = req_bounds; n_target = n_req; restaged = 1; }
        else if (n_bounds != n_req || memcmp(bounds, req_bounds, sizeof(int64_t) * n_req) != 0) { fclose(f); return NULL; }
    }
    int t_stages = n_target + 1;
    /* default: keep the file's mode; a TC file given new stages gets TC on all of them */
    int t_tc = tc_req >= 0 ? tc_req : (restaged && f_tc_stages > 0 ? t_stages : f_tc_stages);
    if (t_tc > t_stages) t_tc = t_stages;
    int t_flags = n_target > 0 && (f_flags || f_tc_stages > 0 || restaged);
    net_t *net = net_alloc(&orig[0][0], len, K, t_tc, t_flags, target, n_target);
    if (!net) { fclose(f); return NULL; }
    int old = v <= 5;                                  /* 4-bit tables: re-index while reading */
    size_t maxfn = 0;
    for (int k = 0; k < K; k++) {
        size_t fn = old ? ((size_t)1 << (4 * len[k])) : net->size[k];
        if (fn > maxfn) maxfn = fn;
    }
    float *tmp = (float *)malloc(maxfn * sizeof(float));
    uint8_t *tmpb = (uint8_t *)malloc(maxfn / 8 + 1);
    for (int st = 0; st < file_stages; st++) {
        for (int k = 0; k < K; k++) {
            size_t fn = old ? ((size_t)1 << (4 * len[k])) : net->size[k];
            int f_has_e = v >= 5 ? st < f_tc_stages : f_e_all;
            int f_has_a = v >= 5 ? st < f_tc_stages : f_a_all;
            int f_has_bits = v >= 5 ? (f_flags && st > 0 && st >= f_tc_stages) : (f_bits_v4 && st > 0);
            if (read_table(f, net->w[st][k], fn, old, len[k], tmp)) goto fail;
            if (f_has_e && read_table(f, stage_is_tc(net, st) ? net->e[st][k] : NULL, fn, old, len[k], tmp)) goto fail;
            if (f_has_a) {
                if (stage_is_tc(net, st)) { if (read_table(f, net->a[st][k], fn, old, len[k], tmp)) goto fail; }
                else if (stage_has_bits(net, st)) {   /* keep the touched flags as bits */
                    if (fread(tmp, sizeof(float), fn, f) != fn) goto fail;
                    for (uint32_t i = 0; i < fn; i++)
                        if (tmp[i] != 0.0f) { uint32_t j = old ? remap_old_index(i, len[k]) : i; net->bits[st][k][j >> 3] |= (uint8_t)(1 << (j & 7)); }
                } else if (skip_bytes(f, fn * sizeof(float))) goto fail;
            }
            if (f_has_bits && read_bits(f, stage_has_bits(net, st) ? net->bits[st][k] : NULL, fn, old, len[k], tmpb)) goto fail;
        }
    }
    free(tmp);
    free(tmpb);
    fclose(f);
    return net;
fail:
    free(tmp);
    free(tmpb);
    fclose(f);
    nt_free(net);
    return NULL;
}

void *nt_load(const char *path) { return nt_load_ex(path, NULL, 0, -1); }
