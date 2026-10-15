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
#if defined(__AVX2__) && defined(__BMI2__)
#include <immintrin.h>
#define NT_SIMD 1                /* x86-64 with AVX2: the 8 symmetric samples of a pattern are indexed at once */
#endif

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
#define TT_BITS 20               /* default transposition table: 2^20 entries (32 MB) per search thread */
#define TT_BITS_MAX 28

static inline int cell(board_t b, int i) { return (int)((b >> (CB * i)) & CMASK); }
static inline board_t put(int v, int i) { return (board_t)v << (CB * i); }
static inline board_t mk(uint64_t lo, uint64_t hi) { return ((board_t)hi << 64) | lo; }
static inline uint64_t lo64(board_t b) { return (uint64_t)b; }
static inline uint64_t hi64(board_t b) { return (uint64_t)(b >> 64); }
static inline uint64_t hash128(board_t b) {
    return (lo64(b) * 0x9E3779B97F4A7C15ULL) ^ (hi64(b) * 0xD1B54A32D192ED03ULL + 0x632BE59BD9B4E019ULL);
}

/* ---------------------------------------------------------------- engine --- */

typedef struct { row_t left, right; int32_t reward_l, reward_r; } rowent_t;   /* one line of cache per row */
static rowent_t *rowtab;
static int tables_ready = 0;
static pthread_mutex_t tables_lock = PTHREAD_MUTEX_INITIALIZER;

static inline row_t reverse_row(row_t r) {
    return ((r >> 15) & 0x1F) | (((r >> 10) & 0x1F) << 5) | (((r >> 5) & 0x1F) << 10) | ((r & 0x1F) << 15);
}

static void init_tables(void) {
    pthread_mutex_lock(&tables_lock);
    if (!tables_ready) {
        rowtab = (rowent_t *)malloc(sizeof(rowent_t) * ROW_N);
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
            rowtab[idx].left = res;
            rowtab[idx].reward_l = reward;
            rowtab[reverse_row(idx)].right = reverse_row(res);
            rowtab[reverse_row(idx)].reward_r = reward;
        }
        tables_ready = 1;
    }
    pthread_mutex_unlock(&tables_lock);
}

/* Two rounds of masked swaps: cells 1-4, 3-6, 9-12, 11-14 (within each 2x2 block),
 * then the top-right 2x2 block with the bottom-left one. */
static inline board_t transpose(board_t x) {
    const board_t m1 = put(CMASK, 1) | put(CMASK, 3) | put(CMASK, 9) | put(CMASK, 11);
    const board_t m2 = put(CMASK, 2) | put(CMASK, 3) | put(CMASK, 6) | put(CMASK, 7);
    board_t t = ((x >> (3 * CB)) ^ x) & m1;
    x ^= t ^ (t << (3 * CB));
    t = ((x >> (6 * CB)) ^ x) & m2;
    return x ^ t ^ (t << (6 * CB));
}

static inline board_t slide_left(board_t b, int *reward) {
    board_t r = 0;
    int s = 0;
    for (int i = 0; i < 4; i++) {
        row_t row = (row_t)((b >> (ROW_BITS * i)) & ROW_MASK);
        r |= (board_t)rowtab[row].left << (ROW_BITS * i);
        s += rowtab[row].reward_l;
    }
    *reward = s;
    return r;
}

static inline board_t slide_right(board_t b, int *reward) {
    board_t r = 0;
    int s = 0;
    for (int i = 0; i < 4; i++) {
        row_t row = (row_t)((b >> (ROW_BITS * i)) & ROW_MASK);
        r |= (board_t)rowtab[row].right << (ROW_BITS * i);
        s += rowtab[row].reward_r;
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
#ifdef NT_SIMD
    /* the empty cells as a bit mask, and the k-th of them by depositing bit k into it */
    __m128i cells = _mm_set_epi64x((long long)_pdep_u64((uint64_t)(b >> 40), 0x1F1F1F1F1F1F1F1FULL),
                                   (long long)_pdep_u64((uint64_t)b, 0x1F1F1F1F1F1F1F1FULL));
    unsigned free_mask = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(cells, _mm_setzero_si128()));
    if (!free_mask) return b;
    uint64_t r0 = rng_next(rng);
    unsigned k0 = (unsigned)((r0 & 0xFFFFFFFFULL) % (uint64_t)__builtin_popcount(free_mask));
    return b | put(((r0 >> 32) % 10 == 0) ? 2 : 1, __builtin_ctz(_pdep_u32(1u << k0, free_mask)));
#else
    int empty = count_empty(b);
    if (!empty) return b;
    uint64_t r = rng_next(rng);
    int k = (int)((r & 0xFFFFFFFFULL) % (uint64_t)empty);
    int val = ((r >> 32) % 10 == 0) ? 2 : 1;
    for (int i = 0; i < 16; i++) {
        if (cell(b, i) == 0 && k-- == 0) return b | put(val, i);
    }
    return b;
#endif
}

static inline uint64_t game_seed(uint64_t seed, long g) {
    return seed * 0x9E3779B97F4A7C15ULL + (uint64_t)g * 0xD1B54A32D192ED03ULL + 0x632BE59BD9B4E019ULL;
}

/* --------------------------------------------------------------- network --- */

typedef struct {
    board_t key;
    double val;
    uint8_t depth;
} tt_entry_t;

typedef struct search_ctx {
    tt_entry_t *tt;
    int bits;                /* the table holds 2^bits entries */
    double cutoff;
    struct net *net;
    uint64_t version;
} search_ctx_t;

/* Stages: a board's stage is the number of boundaries its tile set has reached
 * (see board_scale); each stage owns a table set. The first `tc_stages` stages learn with temporal
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
    uint8_t shuf[MAX_PATTERNS][MAX_LEN][16];   /* shuf[k][j][s] = cells[k][s][j]: byte shuffles for the vector path */
    uint8_t pair[MAX_PATTERNS][48];            /* 6-cell patterns: cells (0,1), (2,3), (4,5) of each symmetry, side by side */
    size_t size[MAX_PATTERNS];
    /* One table per stage and pattern. A plain stage's table is its weights. A temporal coherence stage's
     * table holds a record of four floats per entry: weight, sum of errors, sum of absolute errors, unused;
     * an update then touches one line of cache, not three. See stage_shift, ENTRY_W, ENTRY_E, ENTRY_A. */
    float *w[MAX_STAGES][MAX_PATTERNS];
    uint8_t *bits[MAX_STAGES][MAX_PATTERNS];
    uint64_t version;        /* bumped whenever weights change; search contexts flush on mismatch */
    search_ctx_t *main_ctx;  /* for single-threaded best_move calls */
    search_ctx_t *choose_ctx; /* same, for the choose-mode search (its own transposition table) */
    double leaf_snake;       /* weight of snake_score added to expectimax leaves (0 = tables only) */
    double snake_decay;      /* weight ratio between consecutive cells on the snake path */
    double snake_w[16];      /* snake_decay^k */
    int downgrade;           /* exponent of the tile that switches tile downgrading on at the search root (0 = off) */
    int endgame;             /* exact endgame tables choose the move from this many locked tiles on (0 = off) */
    int expect;              /* training target: 0 = the tile that appeared, 1 = the average over every tile that could */
    int tt_bits;             /* transposition table size for searches on this network */
    int depth_cap[17];       /* random-spawn search: deepest search for a root with this many empty cells (0 = no cap) */
} net_t;

static inline size_t tt_hash(const search_ctx_t *c, board_t b) { return (size_t)(hash128(b) >> (64 - c->bits)); }
#define LEAF_BITS 12             /* leaves get a small table of their own, after the main one in the same block:
                                  * 4,096 entries stay in the core's cache and catch as many repeats as a million do */
static inline tt_entry_t *leaf_slot(const search_ctx_t *c, board_t b) {
    return c->tt + ((size_t)1 << c->bits) + (size_t)(hash128(b) >> (64 - LEAF_BITS));
}

/* Which tiles are on the board: the sum of the distinct tile values present. A
 * boundary is compared with it, so 16384 means "holds a 16384 or larger" and
 * 49152 "holds a 32768 and a 16384, or larger" (the convention of Guei et al.). */
static inline int64_t board_scale(board_t b) {
    int64_t m = 0;
    for (int i = 0; i < 16; i++) {
        int v = (int)cell(b, i);
        if (v) m |= (int64_t)1 << v;
    }
    return m;
}

static inline int stage_of(board_t b, const int64_t *bounds, int n_bounds) {
    if (n_bounds == 0) return 0;
    int64_t m = board_scale(b);
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
 * 15 14 13 12, the k-th weighted by decay^k, best of the 8 symmetries. A chain
 * laid out as a snake scores highest; a scattered chain loses the weight of every
 * tile that is off its place. Added to the search's leaf value with a tunable
 * weight so the chain stays in a shape that can be collapsed. The decay belongs to
 * the network (0.5 by default: the head dominates; nearer 1 the whole chain's
 * order counts). */
#define SNAKE_DECAY_DEFAULT 0.5
static const int SNAKE_PATH[16] = {0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12};
static int snake_cells[8][16];
static pthread_once_t snake_once = PTHREAD_ONCE_INIT;

static void snake_init(void) {
    for (int sym = 0; sym < 8; sym++)
        for (int k = 0; k < 16; k++) snake_cells[sym][k] = sym_cell(SNAKE_PATH[k], sym);
}

static void snake_weights(double decay, double *w) {
    for (int k = 0; k < 16; k++) w[k] = pow(decay, k);
}

static double snake_score(const double *w, board_t b) {
    pthread_once(&snake_once, snake_init);
    double best = 0.0;
    for (int sym = 0; sym < 8; sym++) {
        double acc = 0.0;
        for (int k = 0; k < 16; k++) {
            int v = cell(b, snake_cells[sym][k]);
            if (v) acc += (double)(1u << v) * w[k];
        }
        if (acc > best) best = acc;
    }
    return best;
}

double nt_snake_score(uint64_t lo, uint64_t hi, double decay) {
    double w[16];
    snake_weights(decay, w);
    return snake_score(w, mk(lo, hi));
}

static inline int stage_is_tc(const net_t *net, int st) { return st < net->tc_stages; }
/* Entry i of a stage's table starts at float i << stage_shift: records of 4 floats with temporal coherence. */
static inline int stage_shift(const net_t *net, int st) { return stage_is_tc(net, st) ? 2 : 0; }
#define ENTRY_W(net, st, k, i) ((net)->w[st][k][(size_t)(i) << stage_shift(net, st)])
#define ENTRY_E(net, st, k, i) ((net)->w[st][k][((size_t)(i) << 2) + 1])      /* temporal coherence stages only */
#define ENTRY_A(net, st, k, i) ((net)->w[st][k][((size_t)(i) << 2) + 2])
static inline int stage_has_bits(const net_t *net, int st) { return net->flags && st > 0 && !stage_is_tc(net, st); }
static inline size_t bits_bytes(size_t entries) { return (entries + 7) / 8; }

void nt_free(void *p);

/* The tables are gigabytes of random access. Where Linux gives huge pages only on
 * request, ask for them: far fewer TLB misses per lookup. A hint; failure is harmless. */
static void *table_calloc(size_t n, size_t size) {
    void *p = calloc(n, size);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    const uintptr_t huge = (uintptr_t)2 << 20;
    uintptr_t lo = ((uintptr_t)p + huge - 1) & ~(huge - 1), hi = ((uintptr_t)p + n * size) & ~(huge - 1);
    if (p && hi > lo) madvise((void *)lo, hi - lo, MADV_HUGEPAGE);
#endif
    return p;
}

static net_t *net_alloc(const int *cells_flat, const int *lens, int K, int tc_stages, int flags,
                        const int64_t *bounds, int n_bounds) {
    init_tables();
    if (K < 1 || K > MAX_PATTERNS || n_bounds < 0 || n_bounds >= MAX_STAGES) return NULL;
    net_t *net = (net_t *)calloc(1, sizeof(net_t));
    if (!net) return NULL;
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
        if (n < 1 || n > 7) { nt_free(net); return NULL; }   /* 18^7 entries is the most that fits an index */
        net->len[k] = n;
        for (int j = 0; j < n; j++) {
            int c = cells_flat[k * MAX_LEN + j];
            if (c < 0 || c > 15) { nt_free(net); return NULL; }
            net->orig[k][j] = c;
        }
        for (int s = 0; s < 8; s++)
            for (int j = 0; j < n; j++) net->cells[k][s][j] = sym_cell(net->orig[k][j], s);
        for (int j = 0; j < n; j++)
            for (int s = 0; s < 16; s++) net->shuf[k][j][s] = s < 8 ? (uint8_t)net->cells[k][s][j] : 0x80;
        if (n == 6)
            for (int s = 0; s < 8; s++)
                for (int j = 0; j < 6; j++) net->pair[k][16 * (j / 2) + 2 * s + (j & 1)] = (uint8_t)net->cells[k][s][j];
        net->size[k] = 1;
        for (int j = 0; j < n; j++) net->size[k] *= RADIX;
        for (int st = 0; st < net->n_stages; st++) {
            int ok = (net->w[st][k] = (float *)table_calloc(net->size[k] << stage_shift(net, st), sizeof(float))) != NULL;
            if (stage_has_bits(net, st))
                ok &= (net->bits[st][k] = (uint8_t *)table_calloc(bits_bytes(net->size[k]), 1)) != NULL;
            if (!ok) { nt_free(net); return NULL; }           /* out of memory: the tables are large */
        }
    }
    net->snake_decay = SNAKE_DECAY_DEFAULT;
    snake_weights(net->snake_decay, net->snake_w);
    net->version = 1;
    net->tt_bits = TT_BITS;
    return net;
}

void *nt_create_ex(const int *cells_flat, const int *lens, int K, int tc_stages,
                   const int64_t *bounds, int n_bounds) {
    return net_alloc(cells_flat, lens, K, tc_stages, 1, bounds, n_bounds);
}

/* One network per stage, from n single-stage networks with the same patterns:
 * nets[s] becomes stage s, with nothing read through from an earlier stage.
 * Weights only; this is for play, each stage having been trained on its own. */
void *nt_stack(void **nets, int n, const int64_t *bounds) {
    if (n < 1 || n > MAX_STAGES) return NULL;
    net_t *first = (net_t *)nets[0];
    for (int s = 0; s < n; s++) {
        net_t *x = (net_t *)nets[s];
        if (x->n_stages != 1 || x->K != first->K || memcmp(x->len, first->len, sizeof(first->len)) != 0 ||
            memcmp(x->orig, first->orig, sizeof(first->orig)) != 0) return NULL;
    }
    net_t *net = net_alloc(&first->orig[0][0], first->len, first->K, 0, 0, bounds, n - 1);
    if (!net) return NULL;
    for (int s = 0; s < n; s++)
        for (int k = 0; k < net->K; k++)
            for (size_t i = 0; i < net->size[k]; i++) ENTRY_W(net, s, k, i) = ENTRY_W((net_t *)nets[s], 0, k, i);
    return net;
}

/* Optimistic initialisation: every weight of every stage starts at v, so a board
 * the network has not played is worth 8K * v and the greedy policy is drawn to
 * it until its real value has been learned. Call it on a fresh network. */
void nt_fill(void *p, float v) {
    net_t *net = (net_t *)p;
    for (int st = 0; st < net->n_stages; st++)
        for (int k = 0; k < net->K; k++)
            for (size_t i = 0; i < net->size[k]; i++) ENTRY_W(net, st, k, i) = v;
    net->version++;
}

void nt_free(void *p) {
    net_t *net = (net_t *)p;
    if (!net) return;
    for (int st = 0; st < net->n_stages; st++)
        for (int k = 0; k < net->K; k++) {
            free(net->w[st][k]); free(net->bits[st][k]);
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
            size_t bytes = (net->size[k] << stage_shift(net, st)) * sizeof(float);
            if (net->w[st][k] && mlock(net->w[st][k], bytes) != 0) rc = -1;
            if (net->bits[st][k] && mlock(net->bits[st][k], bits_bytes(net->size[k])) != 0) rc = -1;
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
    if (stage_is_tc(net, st)) return ENTRY_A(net, st, k, i) != 0.0f;
    return (net->bits[st][k][i >> 3] >> (i & 7)) & 1;
}

static inline int entry_source(const net_t *net, int st, int k, uint32_t i) {
    while (st > 0 && !touched(net, st, k, i)) st--;
    return st;
}

static inline float entry_value(const net_t *net, int st, int k, uint32_t i) {
    return ENTRY_W(net, entry_source(net, st, k, i), k, i);
}

static inline float net_value_stage(const net_t *net, board_t b, int st) {
    float v = 0.0f;
    for (int k = 0; k < net->K; k++)
        for (int s = 0; s < 8; s++) v += entry_value(net, st, k, tuple_index(b, net->cells[k][s], net->len[k]));
    return v;
}

/* A board's cells as bytes, and from them the table index of every pattern under
 * every symmetry: idx[8k + s], with the entries of stage st requested from memory
 * on the way. Reading a board this way, once, is several times faster than
 * tuple_index per pattern, and gives the same indices. */
static inline void unpack_cells(board_t b, uint8_t *c) {
#ifdef NT_SIMD
    uint64_t h[2] = {_pdep_u64((uint64_t)b, 0x1F1F1F1F1F1F1F1FULL), _pdep_u64((uint64_t)(b >> 40), 0x1F1F1F1F1F1F1F1FULL)};
    memcpy(c, h, 16);
#else
    for (int i = 0; i < 16; i++) c[i] = (uint8_t)cell(b, i);
#endif
}

static inline void cells_indices(const net_t *net, const uint8_t *c, int st, uint32_t *idx) {
    const int sh = stage_shift(net, st);
#ifdef NT_SIMD
    _Static_assert(RADIX == 18, "the shifts below multiply by 18");
    const __m128i bytes = _mm_loadu_si128((const __m128i *)c);
    const __m256i both = _mm256_broadcastsi128_si256(bytes);
    const __m256i w18 = _mm256_set1_epi16(0x0112), w324 = _mm256_set1_epi32(RADIX * RADIX);
    for (int k = 0; k < net->K; k++) {
        __m256i acc = _mm256_setzero_si256();
        if (net->len[k] == 6) {
            /* two cells at a time: a*18 + b for cells (0,1) and (2,3) of the 8 symmetries
             * in one multiply-add, (4,5) in another, then (p0*324 + p1)*324 + p2 */
            __m256i p01 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(both, _mm256_loadu_si256((const __m256i *)net->pair[k])), w18);
            __m128i p2 = _mm_maddubs_epi16(_mm_shuffle_epi8(bytes, _mm_loadu_si128((const __m128i *)(net->pair[k] + 32))),
                                           _mm256_castsi256_si128(w18));
            acc = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(p01));
            acc = _mm256_add_epi32(_mm256_mullo_epi32(acc, w324), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(p01, 1)));
            acc = _mm256_add_epi32(_mm256_mullo_epi32(acc, w324), _mm256_cvtepu16_epi32(p2));
        } else for (int j = 0; j < net->len[k]; j++) {
            __m128i pick = _mm_shuffle_epi8(bytes, _mm_loadu_si128((const __m128i *)net->shuf[k][j]));
            acc = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(acc, 4), _mm256_slli_epi32(acc, 1)),
                                   _mm256_cvtepu8_epi32(pick));
        }
        _mm256_storeu_si256((__m256i *)(idx + 8 * k), acc);
        const float *w = net->w[st][k];
        if (sh) for (int s = 0; s < 8; s++) __builtin_prefetch(&w[(size_t)idx[8 * k + s] << 2]);
        else for (int s = 0; s < 8; s++) __builtin_prefetch(&w[idx[8 * k + s]]);
    }
#else
    int f = 0;
    for (int k = 0; k < net->K; k++) {
        const int n = net->len[k];
        const float *w = net->w[st][k];
        for (int s = 0; s < 8; s++) {
            const int *cells = net->cells[k][s];
            uint32_t i = 0;
            for (int j = 0; j < n; j++) i = i * RADIX + c[cells[j]];
            __builtin_prefetch(&w[(size_t)i << sh]);
            idx[f++] = i;
        }
    }
#endif
}

/* The entries of one stage at these indices, added in order: pattern by pattern, symmetry by symmetry. */
static inline float sum_entries(const net_t *net, int st, const uint32_t *idx) {
    float v = 0.0f;
    int f = 0;
    if (stage_shift(net, st)) {
        for (int k = 0; k < net->K; k++) {
            const float *w = net->w[st][k];
            for (int s = 0; s < 8; s++) v += w[(size_t)idx[f++] << 2];
        }
    } else {
        for (int k = 0; k < net->K; k++) {
            const float *w = net->w[st][k];
            for (int s = 0; s < 8; s++) v += w[idx[f++]];
        }
    }
    return v;
}

/* The value of a board. Without promotion every entry is read from the board's own
 * stage: index once, then sum in the order net_value_stage uses. The same number. */
static inline float net_value(const net_t *net, board_t b) {
    if (net->flags) return net_value_stage(net, b, stage_of(b, net->bounds, net->n_bounds));
    uint8_t c[16];
    uint32_t idx[MAX_PATTERNS * 8];
    unpack_cells(b, c);
    int st = 0;
    if (net->n_bounds) {
        int64_t scale = 0;
        for (int i = 0; i < 16; i++) scale |= (int64_t)1 << c[i];
        while (st < net->n_bounds && (scale & ~(int64_t)1) >= net->bounds[st]) st++;
    }
    cells_indices(net, c, st, idx);
    return sum_entries(net, st, idx);
}

static inline void net_update_stage(net_t *net, board_t b, float delta, float alpha, float alpha_plain, int st) {
    int tc = stage_is_tc(net, st);
    float step = (tc ? alpha : alpha_plain) * delta / (float)(net->K * 8);
    for (int k = 0; k < net->K; k++) {
        for (int s = 0; s < 8; s++) {
            uint32_t i = tuple_index(b, net->cells[k][s], net->len[k]);
            if (st > 0 && !touched(net, st, k, i)) {   /* promotion: start from the previous stage's entry */
                int src = entry_source(net, st - 1, k, i);
                ENTRY_W(net, st, k, i) = ENTRY_W(net, src, k, i);
                if (tc && stage_is_tc(net, src)) {      /* and inherit its adaptive-rate state */
                    ENTRY_E(net, st, k, i) = ENTRY_E(net, src, k, i);
                    ENTRY_A(net, st, k, i) = ENTRY_A(net, src, k, i);
                }
            }
            if (tc) {
                float E = ENTRY_E(net, st, k, i), A = ENTRY_A(net, st, k, i);
                float beta = (A == 0.0f) ? 1.0f : fabsf(E) / A;
                ENTRY_W(net, st, k, i) += step * beta;
                ENTRY_E(net, st, k, i) = E + delta;
                ENTRY_A(net, st, k, i) = A + fabsf(delta);
            } else {
                ENTRY_W(net, st, k, i) += step;
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

/* ----------------------------------------------------------------- threads --- */

/* Run `fn(job)` on `threads` threads and wait. The job is shared: workers pull
 * game indices from its atomic counter, so fast cores simply take more games. */
static void run_workers(void *(*fn)(void *), void *job, int threads) {
    if (threads < 1) threads = 1;
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)threads);
    int started = 0;
    while (tid && started < threads && pthread_create(&tid[started], NULL, fn, job) == 0) started++;
    if (started == 0) fn(job);                       /* no thread could start: do the work here */
    for (int t = 0; t < started; t++) pthread_join(tid[t], NULL);
    free(tid);
}

/* ---------------------------------------------------------------- learning --- */

/* Single-stage fast path. A board's table indices are computed once, from its
 * cells unpacked to bytes, and kept: the afterstate chosen at one step is the one
 * updated at the next, so its indices are reused for the read and the write. The
 * sums run in the same order as net_value / net_update_stage, so a single-threaded
 * run gives the same weights bit for bit. */
#define MAX_FEATS (MAX_PATTERNS * 8)

static inline void feat_indices(const net_t *net, board_t b, uint32_t *idx) {
    uint8_t c[16];
    unpack_cells(b, c);
    cells_indices(net, c, 0, idx);
}

static inline float feat_value(const net_t *net, const uint32_t *idx) { return sum_entries(net, 0, idx); }

static inline void feat_update(net_t *net, const uint32_t *idx, float delta, float alpha, float alpha_plain) {
    int tc = stage_is_tc(net, 0);
    float step = (tc ? alpha : alpha_plain) * delta / (float)(net->K * 8);
    int f = 0;
    if (tc) {
        for (int k = 0; k < net->K; k++) {
            float *w = net->w[0][k];
            for (int s = 0; s < 8; s++) {
                float *t = &w[(size_t)idx[f++] << 2];          /* weight, error sum, absolute error sum */
                float E = t[1], A = t[2];
                float beta = (A == 0.0f) ? 1.0f : fabsf(E) / A;
                t[0] += step * beta;
                t[1] = E + delta;
                t[2] = A + fabsf(delta);
            }
        }
    } else {
        for (int k = 0; k < net->K; k++) {
            float *w = net->w[0][k];
            for (int s = 0; s < 8; s++) w[idx[f++]] += step;
        }
    }
}

/* The expected-value target (Wang and Matsuzaki): what an afterstate is worth
 * averaged over every tile that can appear on it, a 2 nine times in ten, each
 * followed by the best move's reward and value (nothing when no move is left).
 * Plain TD samples one of these outcomes per update; this uses them all, at the
 * price of up to 30 times the lookups. */
static float expected_target(const net_t *net, board_t after) {
    int empty = 0;
    double sum = 0.0;
    board_t seen_key[256];                               /* values already read in this call; no board is 0 after a move */
    float seen_val[256];
    memset(seen_key, 0, sizeof seen_key);
    for (int i = 0; i < 16; i++) {
        if (cell(after, i)) continue;
        empty++;
        for (int v = 1; v <= 2; v++) {
            board_t s = after | put(v, i);
            float best = 0.0f;
            int any = 0;
            if (!net->flags && !net->n_bounds) {     /* one stage: index the four afterstates, then read them */
                board_t a[4];
                int r[4];
                uint32_t idx[4][MAX_PATTERNS * 8];
                int slot[4], hit[4];
                float hv[4];
                for (int d = 0; d < 4; d++) {
                    a[d] = do_move(s, d, &r[d]);
                    if (a[d] == s) continue;
                    slot[d] = (int)(hash128(a[d]) >> 56);            /* spawns in one line often slide to the same board */
                    hit[d] = seen_key[slot[d]] == a[d];
                    if (hit[d]) hv[d] = seen_val[slot[d]];           /* read now: a later miss may take the slot */
                    else feat_indices(net, a[d], idx[d]);
                }
                for (int d = 0; d < 4; d++) {
                    if (a[d] == s) continue;
                    float va;
                    if (hit[d]) va = hv[d];
                    else { va = feat_value(net, idx[d]); seen_key[slot[d]] = a[d]; seen_val[slot[d]] = va; }
                    float q = (float)r[d] + va;
                    if (!any || q > best) { best = q; any = 1; }
                }
            } else
            for (int d = 0; d < 4; d++) {
                int r;
                board_t a = do_move(s, d, &r);
                if (a == s) continue;
                float q = (float)r + net_value(net, a);
                if (!any || q > best) { best = q; any = 1; }
            }
            sum += (v == 1 ? 0.9 : 0.1) * (double)best;
        }
    }
    return empty ? (float)(sum / empty) : 0.0f;
}

float nt_expected_target(void *p, uint64_t lo, uint64_t hi) { return expected_target((net_t *)p, mk(lo, hi)); }

/* Not for use while games are being played on other threads. Single-stage networks only. */
void nt_set_expect(void *p, int on) { ((net_t *)p)->expect = on != 0; }
int nt_get_expect(void *p) { return ((net_t *)p)->expect; }

/* start: the board a game begins from, or 0 for a new game (two random tiles). */
static void learn_game_single(net_t *net, board_t start, float alpha, float alpha_plain, uint64_t *rng, int64_t *score, int32_t *maxtile, int32_t *nmoves) {
    const size_t fbytes = sizeof(uint32_t) * (size_t)net->K * 8;
    uint32_t idx[4][MAX_FEATS], pidx[MAX_FEATS];
    board_t b = start ? start : spawn(spawn(0, rng), rng);
    int have_prev = 0;
    board_t pafter = 0;
    int64_t sc = 0;
    int32_t mv = 0;
    for (;;) {
        board_t after[4];
        int r[4];
        for (int d = 0; d < 4; d++) {
            after[d] = do_move(b, d, &r[d]);
            if (after[d] != b) feat_indices(net, after[d], idx[d]);
        }
        int best = -1;
        float bestv = 0.0f, bestval = 0.0f;
        for (int d = 0; d < 4; d++) {
            if (after[d] == b) continue;
            float va = feat_value(net, idx[d]);
            float v = (float)r[d] + va;
            if (best < 0 || v > bestv) { bestv = v; best = d; bestval = va; }
        }
        if (have_prev && net->expect) {          /* every outcome of the previous afterstate, this one included */
            float target = expected_target(net, pafter);
            feat_update(net, pidx, target - feat_value(net, pidx), alpha, alpha_plain);
            if (best < 0) break;
        } else {
            if (best < 0) {
                if (have_prev) feat_update(net, pidx, 0.0f - feat_value(net, pidx), alpha, alpha_plain);
                break;
            }
            if (have_prev) feat_update(net, pidx, (float)r[best] + bestval - feat_value(net, pidx), alpha, alpha_plain);
        }
        memcpy(pidx, idx[best], fbytes);
        pafter = after[best];
        have_prev = 1;
        sc += r[best];
        mv++;
        b = spawn(after[best], rng);
    }
    *score = sc;
    *maxtile = max_exp(b) ? (1 << max_exp(b)) : 0;
    *nmoves = mv;
}

static void learn_game(net_t *net, board_t start, float alpha, float alpha_plain, uint64_t *rng, int64_t *score, int32_t *maxtile, int32_t *nmoves) {
    if (net->n_stages == 1) { learn_game_single(net, start, alpha, alpha_plain, rng, score, maxtile, nmoves); return; }
    board_t b = start ? start : spawn(spawn(0, rng), rng);
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
    long *next, n;   /* next: the shared game counter */
    float alpha, alpha_plain;
    uint64_t seed;
    const board_t *starts;
    long n_starts;
    int64_t *scores;
    int32_t *maxtiles, *moves;
} train_job_t;

static void *train_worker(void *arg) {
    train_job_t *j = (train_job_t *)arg;
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        board_t start = j->n_starts > 0 ? j->starts[rng_next(&rng) % (uint64_t)j->n_starts] : 0;
        learn_game(j->net, start, j->alpha, j->alpha_plain, &rng, &j->scores[g], &j->maxtiles[g], &j->moves[g]);
    }
    return NULL;
}

static board_t *unpack_boards(const uint64_t *lohi, long *n) {
    board_t *b = (*n > 0 && lohi) ? (board_t *)malloc(sizeof(board_t) * (size_t)*n) : NULL;
    if (!b) { *n = 0; return NULL; }
    for (long i = 0; i < *n; i++) b[i] = mk(lohi[2 * i], lohi[2 * i + 1]);
    return b;
}

/* starts_lohi: n_starts boards as (lo, hi) pairs. When given, every game begins
 * from one of them, drawn at random, and its score counts from there. This is how
 * a later stage is trained on its own: from boards an earlier network reached. */
void nt_train_from(void *p, long games, int threads, float alpha, float alpha_plain, uint64_t seed,
                   const uint64_t *starts_lohi, long n_starts,
                   int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    net_t *net = (net_t *)p;
    board_t *starts = unpack_boards(starts_lohi, &n_starts);
    long next = 0;
    train_job_t job = {net, &next, games, alpha, alpha_plain, seed, starts, n_starts, scores, maxtiles, moves};
    run_workers(train_worker, &job, threads);
    free(starts);
    net->version++;
}

void nt_train(void *p, long games, int threads, float alpha, float alpha_plain, uint64_t seed,
              int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    nt_train_from(p, games, threads, alpha, alpha_plain, seed, NULL, 0, scores, maxtiles, moves);
}

/* ------------------------------------------------------------- expectimax --- */

/* use_tt = 0 gives a context without a transposition table, for searches made
 * while the weights are being updated (cached values would go stale at once). */
static search_ctx_t *ctx_create_tt(net_t *net, double cutoff, int use_tt) {
    search_ctx_t *c = (search_ctx_t *)calloc(1, sizeof(search_ctx_t));
    c->bits = net->tt_bits;
    c->tt = use_tt ? (tt_entry_t *)table_calloc(((size_t)1 << c->bits) + ((size_t)1 << LEAF_BITS), sizeof(tt_entry_t)) : NULL;
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
        if (c->tt) memset(c->tt, 0, (((size_t)1 << c->bits) + ((size_t)1 << LEAF_BITS)) * sizeof(tt_entry_t));
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

/* Random-spawn search leaves: the tables, plus the snake-order bonus when the net
 * has a leaf weight (a heuristic player on empty tables). */
static inline double leaf_value(const net_t *net, board_t a) {
    double v = net_value(net, a);
    if (net->leaf_snake > 0.0) v += net->leaf_snake * snake_score(net->snake_w, a);
    return v;
}

/* Both setters bump the version: values cached by a search are stale afterwards.
 * Call them between searches, not while games are being played on other threads. */
void nt_set_snake_decay(void *p, double d) {
    net_t *net = (net_t *)p;
    if (net->snake_decay == d) return;
    net->snake_decay = d;
    snake_weights(d, net->snake_w);
    net->version++;
}

double nt_snake_decay(void *p) { return ((net_t *)p)->snake_decay; }

void nt_set_leaf_snake(void *p, double w) {
    net_t *net = (net_t *)p;
    if (w < 0.0) w = 0.0;
    if (net->leaf_snake == w) return;
    net->leaf_snake = w;
    net->version++;
}

double nt_leaf_snake(void *p) { return ((net_t *)p)->leaf_snake; }

/* The table keeps values exactly as computed (doubles), so a hit returns what the
 * search would have found again and, with no probability cutoff, the move chosen
 * never depends on what the table happens to hold: play is the same for any table
 * size or thread count.
 * Leaves are kept too (as depth 0, stored +1 so an empty slot matches nothing):
 * one probe is far cheaper than a board's table lookups. */
/* A max node whose children are all leaves: the four afterstates are probed, indexed
 * and prefetched together, then summed, each in net_value's order. The same numbers. */
static double max_leaf(search_ctx_t *c, board_t b) {
    const net_t *net = c->net;
    if (net->flags || net->leaf_snake > 0.0) return max_node(c, b, 0, 1.0);
    board_t a[4];
    int r[4], st[4], hit[4];
    double val[4];
    tt_entry_t *e[4];
    uint32_t idx[4][MAX_PATTERNS * 8];
    for (int d = 0; d < 4; d++) {
        a[d] = do_move(b, d, &r[d]);
        e[d] = (a[d] != b && c->tt) ? leaf_slot(c, a[d]) : NULL;
        if (e[d]) __builtin_prefetch(e[d]);
    }
    for (int d = 0; d < 4; d++) {
        hit[d] = 0;
        if (a[d] == b) continue;
        if (e[d] && e[d]->key == a[d] && e[d]->depth == 1) { hit[d] = 1; val[d] = e[d]->val; continue; }
        uint8_t cc[16];
        unpack_cells(a[d], cc);
        int s0 = 0;
        if (net->n_bounds) {
            int64_t scale = 0;
            for (int i = 0; i < 16; i++) scale |= (int64_t)1 << cc[i];
            while (s0 < net->n_bounds && (scale & ~(int64_t)1) >= net->bounds[s0]) s0++;
        }
        st[d] = s0;
        cells_indices(net, cc, s0, idx[d]);
    }
    double best = 0.0;
    int any = 0;
    for (int d = 0; d < 4; d++) {
        if (a[d] == b) continue;
        if (!hit[d]) {
            val[d] = (double)sum_entries(net, st[d], idx[d]);
            if (e[d]) { e[d]->key = a[d]; e[d]->depth = 1; e[d]->val = val[d]; }
        }
        double v = (double)r[d] + val[d];
        if (!any || v > best) { best = v; any = 1; }
    }
    return best;
}

static double chance_node(search_ctx_t *c, board_t after, int depth, double prob) {
    if (prob < c->cutoff) depth = 0;
    tt_entry_t *e = !c->tt ? NULL : depth == 0 ? leaf_slot(c, after) : &c->tt[tt_hash(c, after)];
    if (e && e->key == after && e->depth == depth + 1) return e->val;
    if (depth == 0) {
        double v = leaf_value(c->net, after);
        if (e) { e->key = after; e->depth = 1; e->val = v; }
        return v;
    }
    int empty = count_empty(after);
    double p2 = prob * 0.9 / empty, p4 = prob * 0.1 / empty, sum = 0.0;
    for (int i = 0; i < 16; i++) {
        if (cell(after, i) != 0) continue;
        if (depth == 1) {
            sum += 0.9 * max_leaf(c, after | (put(1, i)));
            sum += 0.1 * max_leaf(c, after | (put(2, i)));
            continue;
        }
        sum += 0.9 * max_node(c, after | (put(1, i)), depth - 1, p2);
        sum += 0.1 * max_node(c, after | (put(2, i)), depth - 1, p4);
    }
    double v = sum / empty;
    if (e) {
        e->key = after;
        e->depth = (uint8_t)(depth + 1);
        e->val = v;
    }
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

/* Tile downgrading (Guei et al. 2022): halve every tile above the largest tile
 * value that is missing below the board's maximum. The gap closes, every merge
 * stays possible and no new one appears, and the board lands where the tables
 * have seen far more training. Used at the search root only, for choosing the
 * move, once the board holds the network's threshold tile. */
static board_t downgrade(board_t b) {
    unsigned present = 0;
    for (int i = 0; i < 16; i++) present |= 1u << cell(b, i);
    int gap = max_exp(b) - 1;
    while (gap >= 1 && ((present >> gap) & 1)) gap--;
    if (gap < 1) return b;
    board_t out = 0;
    for (int i = 0; i < 16; i++) {
        int v = cell(b, i);
        out |= put(v > gap ? v - 1 : v, i);
    }
    return out;
}

void nt_downgrade(uint64_t lo, uint64_t hi, uint64_t *out_lo, uint64_t *out_hi) {
    board_t r = downgrade(mk(lo, hi));
    *out_lo = lo64(r);
    *out_hi = hi64(r);
}

/* Not for use while games are being played on other threads. */
void nt_set_downgrade(void *p, int exp) { ((net_t *)p)->downgrade = exp < 0 ? 0 : exp; }
int nt_get_downgrade(void *p) { return ((net_t *)p)->downgrade; }

/* Not for use while games are being played on other threads. bits: 2^bits entries
 * of 32 bytes per search thread (plus the 2^LEAF_BITS entries kept for leaves). caps: 17 depths indexed by the root's empty cells
 * (0 = no cap), or NULL to clear: crowded boards get the deep search, open ones,
 * where a spawn rarely decides anything, a shallow one. */
static void drop_ctx(search_ctx_t **c) {
    if (*c) { free((*c)->tt); free(*c); *c = NULL; }
}

void nt_set_tt_bits(void *p, int bits) {
    net_t *net = (net_t *)p;
    if (bits < 10) bits = 10;
    if (bits > TT_BITS_MAX) bits = TT_BITS_MAX;
    if (bits == net->tt_bits) return;
    net->tt_bits = bits;
    drop_ctx(&net->main_ctx);                            /* rebuilt at the new size on the next search */
    drop_ctx(&net->choose_ctx);
}

int nt_get_tt_bits(void *p) { return ((net_t *)p)->tt_bits; }

void nt_set_depth_caps(void *p, const int *caps) {
    net_t *net = (net_t *)p;
    for (int i = 0; i < 17; i++) net->depth_cap[i] = caps && caps[i] > 0 ? caps[i] : 0;
}

void nt_get_depth_caps(void *p, int *out) {
    for (int i = 0; i < 17; i++) out[i] = ((net_t *)p)->depth_cap[i];
}

/* ------------------------------------------------------------- endgame --- */

/* Exact tables for the end of the game, used to choose the move once a board qualifies.
 *
 * A board qualifies with k locked tiles when, in one of its eight orientations, 32768, 16384, ... down to
 * 32768 >> (k-1) sit one each on the first k cells of the snake (0 1 2 3, 7 6 5 4, 8 9 10 11, 15 14 13 12) and
 * every other tile is at most the last of them. The n = 16 - k other cells are in play. A move is allowed if
 * it leaves the locked cells as they are and makes no tile in play larger than the last locked one; merging a
 * tile into the last locked one wins (the ladder then merges upward, counted as certain); a tile of half the
 * last locked one on the next snake cell may be taken as locked too (the better of the two is used).
 * Value = chance of winning with the best allowed move each turn, solved by recursion with a memo (every move
 * adds a tile, so there are no cycles). Restricting the moves makes it a floor on perfect play.
 *
 * The memo is shared by every network and thread, filled on demand under one lock. Tables with 8 or more
 * locked tiles are arrays of every position; larger ones keep only the positions reached, 7 bytes each (the
 * index in 40 bits, the value in 16: steps of 1/65535), in maps of 2^EG_BITS slots (2^(EG_BITS-2) for 7 locked).
 * Values are what the memo returns, so play is the same whatever order positions were first reached in. */
#define EG_MIN 5
/* Formations ("families"): each is an order of cells. The k largest tiles lie on its first k cells, the next rung
 * goes on cell k and the other cells are in play. Family 0 is the snake: the only one solved on demand. Others
 * come only from tables built in advance (nt_endgame_load with their path). */
#define EG_FAM 4
static int eg_path[EG_FAM][16] = {{0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12}};
static int eg_nfam = 1;
/* eg_kvalid[f]: tables of family f are used up to this many locked tiles: the longest prefix of its path in which
 * consecutive cells are neighbours (beyond it a merge into the last locked tile does not collapse the ladder). */
static int eg_kvalid[EG_FAM] = {15};
#define EG_SNAKE eg_path[0]

static int eg_chain(const int *path) {
    int k = 1;
    while (k < 16) {
        int a = path[k - 1], b = path[k], dr = a / 4 - b / 4, dc = a % 4 - b % 4;
        if (dr * dr + dc * dc != 1) break;
        k++;
    }
    return k > 15 ? 15 : k;
}
static pthread_mutex_t eg_lock = PTHREAD_MUTEX_INITIALIZER;
static float *eg_dense[16];
static uint32_t *eg_klo[16]; static uint8_t *eg_khi[16]; static uint16_t *eg_val[16];
/* Maps of 2^31 slots or more use 4 bytes a slot instead: an index is first mixed by a 40-bit bijection h; the
 * slot it starts from holds the low bits of h, so the word only needs the high bits of h (q), how far the entry
 * sits from its starting slot (off) and the value. 0 = empty. Entries that would sit too far are not stored. */
static uint32_t *eg_w32[16]; static int eg_wbits[16];
static uint64_t eg_cap[16], eg_used[16];
static int eg_bits = 29, eg_full = 0;
/* Tables built whole in advance (nt_endgame_load): every position of a capped space, 2 bytes each, mapped read-only
 * from files; no lock needed. The tile on the j-th free snake cell of table k is at most 2^eg_tcap[k][j] (n = 16 - k
 * without caps); a board that does not fit is not a position of that table and a move that breaks a cap is not
 * allowed. Index (layout 2 of notes/tb/build2.c): positions sorted by the sum of the tiles in play (in units of 2),
then by rank within the sum, the first free cell most significant: offs[s] + the sum over cells of eg_rank[k][j]
at (sum left, tile), the number of ways to fill the later cells with what is left when cell j holds less. */
static uint16_t *eg_pre[EG_FAM][16];
static uint64_t *eg_offs[EG_FAM][16], *eg_rank[EG_FAM][16][16], eg_psize[EG_FAM][16];
static int eg_tcap[EG_FAM][16][16];

static inline int eg_units(int e) { return e ? 1 << (e - 1) : 0; }

static inline int eg_fits(int f, const uint8_t *b, int k) {
    const int *P = eg_path[f];
    for (int j = 0; j < 16 - k; j++) if (b[P[k + j]] > eg_tcap[f][k][j]) return 0;
    return 1;
}

static void eg_caps_init(void) {
    for (int f = 0; f < EG_FAM; f++) for (int k = 0; k < 16; k++) for (int j = 0; j < 16; j++) eg_tcap[f][k][j] = 16 - k;
}
static pthread_once_t eg_caps_once = PTHREAD_ONCE_INIT;
static void eg_caps_default(void) { pthread_once(&eg_caps_once, eg_caps_init); }   /* no caps: tiles in play up to n */

static int eg_slide(uint8_t *l) {
    uint8_t t[4]; int n = 0, changed = 0;
    for (int i = 0; i < 4; i++) if (l[i]) t[n++] = l[i];
    uint8_t o[4] = {0, 0, 0, 0}; int m = 0;
    for (int i = 0; i < n;) {
        if (i + 1 < n && t[i] == t[i + 1]) { o[m++] = (uint8_t)(t[i] + 1); i += 2; }
        else o[m++] = t[i++];
    }
    for (int i = 0; i < 4; i++) { if (l[i] != o[i]) changed = 1; l[i] = o[i]; }
    return changed;
}

/* dir: 0 up, 1 right, 2 down, 3 left, as do_move */
static int eg_move(const uint8_t *b, int dir, uint8_t *out) {
    int changed = 0;
    memcpy(out, b, 16);
    for (int i = 0; i < 4; i++) {
        uint8_t l[4]; int c[4];
        for (int j = 0; j < 4; j++)
            c[j] = dir == 0 ? 4 * j + i : dir == 2 ? 4 * (3 - j) + i : dir == 3 ? 4 * i + j : 4 * i + (3 - j);
        for (int j = 0; j < 4; j++) l[j] = b[c[j]];
        changed |= eg_slide(l);
        for (int j = 0; j < 4; j++) out[c[j]] = l[j];
    }
    return changed;
}

static uint64_t eg_index(const uint8_t *b, int k) {
    int n = 16 - k; uint64_t idx = 0;
    for (int i = 15; i >= k; i--) idx = idx * (uint64_t)(n + 1) + b[EG_SNAKE[i]];
    return idx;
}

static uint64_t eg_pow(uint64_t b, int e) { uint64_t r = 1; while (e--) r *= b; return r; }

static int eg_alloc(int k) {                                  /* under eg_lock */
    if (eg_dense[k] || eg_klo[k] || eg_w32[k]) return 0;
    if (k >= 8) {
        uint64_t size = eg_pow((uint64_t)(17 - k), 16 - k);
        float *t = (float *)malloc(size * sizeof(float));
        if (!t) { eg_full = 1; return -1; }
        for (uint64_t i = 0; i < size; i++) t[i] = -1.0f;
        eg_dense[k] = t;
        return 0;
    }
    /* 5 locked: 2^eg_bits slots; 6: at most 2^31; 7: two bits fewer, at most 2^27 (it never needed more) */
    int bits = k == 5 ? eg_bits : k == 6 ? (eg_bits < 31 ? eg_bits : 31) : (eg_bits - 2 < 27 ? eg_bits - 2 : 27);
    uint64_t cap = (uint64_t)1 << bits;
    if (bits >= 31) {
        uint32_t *w = (uint32_t *)calloc(cap, 4);
        if (!w) { eg_full = 1; return -1; }
        eg_cap[k] = cap; eg_wbits[k] = bits; eg_w32[k] = w;
        return 0;
    }
    uint32_t *klo = (uint32_t *)malloc(cap * 4);
    uint8_t *khi = (uint8_t *)malloc(cap);
    uint16_t *val = (uint16_t *)malloc(cap * 2);
    if (!klo || !khi || !val) { free(klo); free(khi); free(val); eg_full = 1; return -1; }   /* all or nothing */
    memset(khi, 0xFF, cap);                                   /* an index never has 0xFF in its top byte */
    eg_cap[k] = cap; eg_klo[k] = klo; eg_khi[k] = khi; eg_val[k] = val;
    return 0;
}

static inline uint64_t eg_slot(int k, uint64_t key) {
    uint64_t x = key; x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    uint64_t m = eg_cap[k] - 1, i = x & m;
    uint32_t lo = (uint32_t)key; uint8_t hi = (uint8_t)(key >> 32);
    while (eg_khi[k][i] != 0xFF && !(eg_klo[k][i] == lo && eg_khi[k][i] == hi)) i = (i + 1) & m;
    return i;
}

static inline uint64_t eg_mix40(uint64_t x) {             /* a bijection on 40 bits */
    const uint64_t M = ((uint64_t)1 << 40) - 1;
    x = (x * 0x9E3779B97F4A7C15ULL) & M; x ^= x >> 20;
    x = (x * 0xC2B2AE3D27D4EB4FULL) & M; x ^= x >> 20;
    return x;
}

/* 4-byte map: word = 1 | off (bits-25 bits, at most 8) | q (40-bits bits) | value (16 bits), from the top.
 * Returns the slot holding key, or the empty slot where it goes, or UINT64_MAX if it would sit too far. */
static inline uint64_t eg_w32_slot(int k, uint64_t key, int *found) {
    int bits = eg_wbits[k], qbits = 40 - bits, obits = 31 - 16 - qbits;
    if (obits > 8) obits = 8;
    uint64_t h = eg_mix40(key), m = eg_cap[k] - 1, home = h & m, q = h >> bits;
    for (uint64_t off = 0; off < ((uint64_t)1 << obits); off++) {
        uint64_t i = (home + off) & m;
        uint32_t w = eg_w32[k][i];
        if (!w) { *found = 0; return i; }
        if (((w >> 16) & (((uint32_t)1 << qbits) - 1)) == q && ((w >> (16 + qbits)) & (((uint32_t)1 << obits) - 1)) == off) {
            *found = 1; return i;
        }
    }
    *found = 0;
    return UINT64_MAX;
}

static inline uint32_t eg_w32_word(int k, uint64_t key, uint64_t slot, uint16_t v) {
    int bits = eg_wbits[k], qbits = 40 - bits;
    uint64_t h = eg_mix40(key), m = eg_cap[k] - 1, off = (slot - (h & m)) & m, q = h >> bits;
    return (uint32_t)1 << 31 | (uint32_t)(off << (16 + qbits)) | (uint32_t)(q << 16) | v;
}

static float eg_value(int f, const uint8_t *b, int k);

/* the expected value of an afterstate in table k, over the tile that comes next */
static float eg_after(int f, uint8_t *a, int k) {
    int n = 16 - k, empty = 0;
    double sum = 0.0;
    for (int c = 0; c < 16; c++) {
        if (a[c]) continue;
        empty++;
        a[c] = 1; sum += 0.9 * eg_value(f, a, k);
        a[c] = 2; sum += n >= 2 ? 0.1 * eg_value(f, a, k) : 0.0;   /* a 4 cannot be in play with one cell: lost */
        a[c] = 0;
    }
    return empty ? (float)(sum / empty) : 0.0f;
}

/* -1: the move is not allowed; 2: it wins; else 0 and *a holds the afterstate */
static int eg_try(int f, const uint8_t *b, int k, int d, uint8_t *a) {
    int n = 16 - k;
    const int *P = eg_path[f];
    if (!eg_move(b, d, a)) return -1;
    for (int i = 0; i < k - 1; i++) if (a[P[i]] != 15 - i) return -1;
    if (a[P[k - 1]] == n + 1) return 2;                          /* merged into the last locked tile */
    if (a[P[k - 1]] != n) return -1;
    if (!eg_fits(f, a, k)) return -1;
    return 0;
}

static float eg_value(int f, const uint8_t *b, int k) {
    int n = 16 - k;
    if (k >= 16) return 1.0f;
    if (eg_pre[f][k]) {
        const int *P = eg_path[f];
        int r = 0;
        for (int j = 0; j < n; j++) r += eg_units(b[P[k + j]]);
        uint64_t q = eg_offs[f][k][r];
        for (int j = 0; j < n; j++) {
            int v = b[P[k + j]];
            q += eg_rank[f][k][j][(size_t)r * (eg_tcap[f][k][j] + 1) + v];
            r -= eg_units(v);
        }
        return (float)eg_pre[f][k][q] * (1.0f / 65535.0f);
    }
    if (f) return 0.0f;                                          /* only the snake is solved on demand */
    uint64_t idx = eg_index(b, k), slot = 0;
    if (eg_alloc(k) != 0) return 0.0f;
    if (eg_dense[k]) { if (eg_dense[k][idx] >= 0.0f) return eg_dense[k][idx]; }
    else if (eg_w32[k]) {
        int found; uint64_t i = eg_w32_slot(k, idx, &found);
        if (found) return (float)(eg_w32[k][i] & 0xFFFF) * (1.0f / 65535.0f);
    } else {
        slot = eg_slot(k, idx);
        if (eg_khi[k][slot] != 0xFF) return (float)eg_val[k][slot] * (1.0f / 65535.0f);
    }
    float best = 0.0f;
    for (int d = 0; d < 4; d++) {
        uint8_t a[16];
        int r = eg_try(0, b, k, d, a);
        if (r < 0) continue;
        if (r == 2) { best = 1.0f; break; }
        float v = eg_after(0, a, k);
        if (v > best) best = v;
    }
    if (k < 15 && b[EG_SNAKE[k]] == n - 1) {                   /* the next tile of the ladder is in place: lock it */
        if (eg_fits(0, b, k + 1)) { float w = eg_value(0, b, k + 1); if (w > best) best = w; }
    }
    if (eg_dense[k]) eg_dense[k][idx] = best;
    else if (eg_w32[k]) {
        int found; uint64_t i = eg_w32_slot(k, idx, &found);    /* again: the map may have changed in the recursion */
        uint16_t q16 = (uint16_t)(best * 65535.0f + 0.5f);
        if (i != UINT64_MAX && !found && eg_used[k] * 10 < eg_cap[k] * 8) {
            eg_w32[k][i] = eg_w32_word(k, idx, i, q16);
            eg_used[k]++;
            best = (float)q16 * (1.0f / 65535.0f);
        } else if (found) best = (float)(eg_w32[k][i] & 0xFFFF) * (1.0f / 65535.0f);
        else eg_full = 1;
    } else if (eg_used[k] * 10 < eg_cap[k] * 8) {
        slot = eg_slot(k, idx);                                 /* the map may have changed during the recursion */
        eg_klo[k][slot] = (uint32_t)idx; eg_khi[k][slot] = (uint8_t)(idx >> 32);
        eg_val[k][slot] = (uint16_t)(best * 65535.0f + 0.5f);
        eg_used[k]++;
        best = (float)eg_val[k][slot] * (1.0f / 65535.0f);
    } else eg_full = 1;
    return best;
}

/* Turns the board by symmetry s for family f: fills o with the turned board, its corner set to 32768 (see
 * eg_choose), and *corner with the real corner, and returns how many tiles lie in place along the path (kmax), or
 * 0 if the corner is not the largest. The board is then a position of every table k <= kmax that it fits: there the
 * tiles from k on are in play. */
static int eg_orient(int f, const uint8_t *c, int s, uint8_t *o, int *corner) {
    const int *P = eg_path[f];
    for (int i = 0; i < 16; i++) o[i] = c[sym_cell(i, s)];
    *corner = o[P[0]];
    if (*corner < 15) return 0;
    o[P[0]] = 15;
    int kmax = 0;
    while (kmax < 15 && o[P[kmax]] == 15 - kmax) kmax++;
    return kmax > eg_kvalid[f] ? eg_kvalid[f] : kmax;
}

/* Whether the turned board o is a position of family f's table k (which exists, with its caps fitting). */
static int eg_usable(int f, const uint8_t *o, int k) {
    if (k < EG_MIN || k > eg_kvalid[f] || (f && !eg_pre[f][k])) return 0;
    eg_caps_default();
    return eg_fits(f, o, k);
}

/* The largest k (at least kmin) for which some orientation of the board is a position of a table of a family in
 * fmask (bit f = family f), or 0. */
static int eg_position(board_t board, int kmin, unsigned fmask) {
    uint8_t c[16], o[16];
    int corner, best = 0;
    for (int i = 0; i < 16; i++) c[i] = (uint8_t)cell(board, i);
    for (int f = 0; f < eg_nfam; f++) {
        if (!((fmask >> f) & 1)) continue;
        for (int s = 0; s < 8; s++) {
            int kmax = eg_orient(f, c, s, o, &corner);
            for (int k = kmax; k >= kmin && k > best; k--) if (eg_usable(f, o, k)) { best = k; break; }
        }
    }
    return best;
}

/* The table's move for this board, or -1 when no orientation of it is a table position with kmin or more locked
 * tiles and a chance above zero. Works on the board as it is (before any downgrading).
 * The corner tile only has to be the largest: a 65536 there with 16384, 8192, ... after it plays exactly like a
 * 32768 there (locked tiles never move or merge in the table), so the same tables serve the game after the first
 * 65536, where winning means making the 32768 beside it. */
static int eg_choose(board_t board, int kmin) {
    uint8_t c[16];
    for (int i = 0; i < 16; i++) c[i] = (uint8_t)cell(board, i);
    int best_dir = -1; float best_v = 0.0f;
    /* every formation, orientation and table the board is a position of: the move with the best chance. A smaller
     * table allows moves a larger one forbids (it may lock the next tile later), but its caps may forbid moves the
     * larger one allows, so neither dominates: all are tried. */
    for (int f = 0; f < eg_nfam; f++)
    for (int s = 0; s < 8; s++) {
        uint8_t o[16];                                          /* the board turned or mirrored */
        int corner, kmax = eg_orient(f, c, s, o, &corner);
        const int *P = eg_path[f];
        for (int k = kmin > EG_MIN ? kmin : EG_MIN; k <= kmax; k++) {
        if (!eg_usable(f, o, k)) continue;
        for (int d = 0; d < 4; d++) {                           /* each move, made on the real board */
            int r;
            board_t after = do_move(board, d, &r);
            if (after == board) continue;
            uint8_t a[16], t[16];
            for (int i = 0; i < 16; i++) a[i] = (uint8_t)cell(after, i);
            for (int i = 0; i < 16; i++) t[i] = a[sym_cell(i, s)];
            if (t[P[0]] != corner) continue;                    /* the corner moved: not an allowed move */
            t[P[0]] = 15;
            int ok2 = 1;                                        /* the same tests as eg_try, on the turned afterstate */
            for (int i = 0; i < k - 1 && ok2; i++) if (t[P[i]] != 15 - i) ok2 = 0;
            if (!ok2) continue;
            float v;
            if (t[P[k - 1]] == 17 - k) v = 1.0f;
            else {
                if (t[P[k - 1]] != 16 - k || !eg_fits(f, t, k)) continue;
                if (eg_pre[f][k]) v = eg_after(f, t, k);        /* read-only: no lock */
                else {
                    pthread_mutex_lock(&eg_lock);
                    v = eg_after(f, t, k);
                    pthread_mutex_unlock(&eg_lock);
                }
            }
            if (v > best_v) { best_v = v; best_dir = d; }
        }
        }
    }
    return best_dir;
}

/* From each start board, play until the board is a table position with k or more locked tiles (reached = 1) or
 * the game is lost (0), with the network's settings (search depth, downgrading; tables are not used). For
 * measuring how often a player gets to the tables. Game i draws its tiles from game_seed(seed, i). */
static int best_move(search_ctx_t *c, board_t b, int depth);

typedef struct {
    net_t *net;
    int depth, k;
    unsigned fmask;
    long *next, n;
    uint64_t seed;
    const board_t *starts;
    int32_t *reached, *moves;
} reach_job_t;

static void *reach_worker(void *arg) {
    reach_job_t *j = (reach_job_t *)arg;
    search_ctx_t *c = ctx_create(j->net, 0.0);
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        board_t b = j->starts[g];
        int32_t mv = 0, got = 0;
        for (;;) {
            if (eg_position(b, j->k, j->fmask)) { got = 1; break; }
            int d = best_move(c, b, j->depth);
            if (d < 0) break;
            int r;
            b = spawn(do_move(b, d, &r), &rng);
            mv++;
        }
        j->reached[g] = got;
        j->moves[g] = mv;
    }
    ctx_free(c);
    return NULL;
}

/* fmask: which formations count (bit f = family f; family 0 = the snake, also without tables) */
void nt_reach(void *p, const uint64_t *starts_lohi, long n, int k, unsigned fmask, int depth, int threads,
              uint64_t seed, int32_t *reached, int32_t *moves) {
    net_t *net = (net_t *)p;
    long nn = n;
    board_t *starts = unpack_boards(starts_lohi, &nn);
    int keep = net->endgame;
    net->endgame = 0;                                       /* the point is to see how the network gets there */
    long next = 0;
    reach_job_t job = {net, depth, k, fmask, &next, nn, seed, starts, reached, moves};
    run_workers(reach_worker, &job, threads);
    net->endgame = keep;
    free(starts);
}

/* From each start board, play to the end with the network's settings (search depth, downgrading, endgame tables):
 * the largest tile reached, the points scored from the start and the moves. For measuring the end of the game
 * many times from late boards instead of whole games. Game i draws its tiles from game_seed(seed, i). */
typedef struct {
    net_t *net;
    int depth;
    long *next, n;
    uint64_t seed;
    const board_t *starts;
    int32_t *maxtile, *moves;
    int64_t *score;
    uint32_t *seen;
} from_job_t;

static void *from_worker(void *arg) {
    from_job_t *j = (from_job_t *)arg;
    search_ctx_t *c = ctx_create(j->net, 0.0);
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        board_t b = j->starts[g];
        int64_t sc = 0;
        int32_t mv = 0;
        uint32_t seen = 0;
        for (;;) {
            for (int i = 0; i < 16; i++) seen |= 1u << cell(b, i);
            int d = best_move(c, b, j->depth);
            if (d < 0) break;
            int r;
            b = spawn(do_move(b, d, &r), &rng);
            sc += r;
            mv++;
        }
        if (j->seen) j->seen[g] = seen;
        j->maxtile[g] = max_exp(b) ? 1 << max_exp(b) : 0;
        j->score[g] = sc;
        j->moves[g] = mv;
    }
    ctx_free(c);
    return NULL;
}

/* seen (may be NULL): bit e set when a tile 2^e was on the board at some point of the game */
void nt_play_boards(void *p, const uint64_t *starts_lohi, long n, int depth, int threads, uint64_t seed,
                    int32_t *maxtile, int64_t *score, int32_t *moves, uint32_t *seen) {
    net_t *net = (net_t *)p;
    long nn = n, next = 0;
    board_t *starts = unpack_boards(starts_lohi, &nn);
    from_job_t job = {net, depth, &next, nn, seed, starts, maxtile, moves, score, seen};
    run_workers(from_worker, &job, threads);
    free(starts);
}

/* Use tables built in advance (endgame_build.c): <dir>/k<k>.u16 for k = kmin .. 15, 2-byte values, along the
 * formation `path` (16 cells; NULL = the snake), with the tile on the j-th free cell of the path capped at
 * 2^(n - drops[j]) (the last drop repeats; never below 4, never above n) in tables with at most kcap locked tiles;
 * larger k uncapped. Maps the files read-only. The snake's tables are family 0 (below kmin they stay on demand and
 * step into these ones); another path becomes a new family, used only where its tables exist. Returns the family,
 * or -1 (nothing changed) if a file is missing or the wrong size, the family is already loaded or there is no room.
 * Call before any play. */
static int eg_family_loaded(int f) { for (int k = EG_MIN; k <= 15; k++) if (eg_pre[f][k]) return 1; return 0; }
int nt_endgame_load_path(const char *dir, int kmin, const int *drops, int ndrops, int kcap, const int *path) {
    if (kmin < EG_MIN || kmin > 15 || ndrops < 0 || ndrops > 16) return -1;
    int fam = 0;
    if (path && memcmp(path, eg_path[0], sizeof eg_path[0]) != 0) {
        for (fam = 1; fam < eg_nfam; fam++) if (memcmp(path, eg_path[fam], sizeof eg_path[0]) == 0) break;
        if (fam == EG_FAM) return -1;
        unsigned seen = 0;                                       /* a path must hold every cell once */
        for (int i = 0; i < 16; i++) { if (path[i] < 0 || path[i] > 15) return -1; seen |= 1u << path[i]; }
        if (seen != 0xFFFF) return -1;
    }
    if (eg_family_loaded(fam)) return -1;
    eg_caps_default();
    int caps[16][16];
    uint64_t *offs[16] = {0}, *rank[16][16] = {{0}}, size[16];
    uint16_t *maps[16] = {0};
    int ok = 1;
    for (int k = kmin; k <= 15 && ok; k++) {
        int n = 16 - k, S = 0;
        for (int j = 0; j < n; j++) {
            int d = ndrops && k <= kcap ? drops[j < ndrops ? j : ndrops - 1] : 0, c = n - d;
            if (c < 2) c = 2;
            if (c > n) c = n;
            caps[k][j] = c; S += eg_units(c);
        }
        uint64_t *cnt = (uint64_t *)calloc((size_t)(n + 1) * (S + 1), sizeof(uint64_t));   /* cells j.. with r units */
        offs[k] = (uint64_t *)malloc((size_t)(S + 2) * sizeof(uint64_t));
        if (!cnt || !offs[k]) { free(cnt); ok = 0; break; }
        cnt[(size_t)n * (S + 1)] = 1;
        for (int j = n - 1; j >= 0; j--)
            for (int r = 0; r <= S; r++) {
                uint64_t t = 0;
                for (int e = 0; e <= caps[k][j] && eg_units(e) <= r; e++) t += cnt[(size_t)(j + 1) * (S + 1) + r - eg_units(e)];
                cnt[(size_t)j * (S + 1) + r] = t;
            }
        offs[k][0] = 0;
        for (int r = 0; r <= S; r++) offs[k][r + 1] = offs[k][r] + cnt[r];
        size[k] = offs[k][S + 1];
        for (int j = 0; j < n && ok; j++) {
            int C = caps[k][j] + 1;
            rank[k][j] = (uint64_t *)malloc((size_t)(S + 1) * C * sizeof(uint64_t));
            if (!rank[k][j]) { ok = 0; break; }
            for (int r = 0; r <= S; r++) {
                uint64_t acc = 0;
                for (int v = 0; v < C; v++) {
                    rank[k][j][(size_t)r * C + v] = acc;
                    if (eg_units(v) <= r) acc += cnt[(size_t)(j + 1) * (S + 1) + r - eg_units(v)];
                }
            }
        }
        free(cnt);
        if (!ok) break;
        char file[4096];
        snprintf(file, sizeof file, "%s/k%d.u16", dir, k);
        FILE *fp = fopen(file, "rb");
        void *m = MAP_FAILED;
        if (fp) {
            fseek(fp, 0, SEEK_END);
            if ((uint64_t)ftell(fp) == size[k] * 2) m = mmap(NULL, size[k] * 2, PROT_READ, MAP_SHARED, fileno(fp), 0);
            fclose(fp);
        }
        if (m == MAP_FAILED) { ok = 0; break; }
        madvise(m, size[k] * 2, MADV_RANDOM);
        maps[k] = (uint16_t *)m;
    }
    if (!ok) {
        for (int k = kmin; k <= 15; k++) {
            if (maps[k]) munmap(maps[k], size[k] * 2);
            free(offs[k]);
            for (int j = 0; j < 16; j++) free(rank[k][j]);
        }
        return -1;
    }
    if (fam && fam == eg_nfam) { memcpy(eg_path[fam], path, sizeof eg_path[0]); eg_kvalid[fam] = eg_chain(path); eg_nfam++; }
    for (int k = kmin; k <= 15; k++) {
        eg_offs[fam][k] = offs[k]; eg_psize[fam][k] = size[k];
        for (int j = 0; j < 16 - k; j++) { eg_tcap[fam][k][j] = caps[k][j]; eg_rank[fam][k][j] = rank[k][j]; }
    }
    for (int k = kmin; k <= 15; k++) eg_pre[fam][k] = maps[k]; /* last: a table is used only once all is set */
    return fam;
}
int nt_endgame_load(const char *dir, int kmin, const int *drops, int ndrops, int kcap) {
    return nt_endgame_load_path(dir, kmin, drops, ndrops, kcap, NULL) < 0 ? -1 : 0;
}
/* The lowest k with a table built in advance for family f, or 0. */
int nt_endgame_loaded_family(int f) {
    if (f < 0 || f >= eg_nfam) return 0;
    for (int k = EG_MIN; k <= 15; k++) if (eg_pre[f][k]) return k;
    return 0;
}
int nt_endgame_loaded(void) { return nt_endgame_loaded_family(0); }
int nt_endgame_families(void) { return eg_nfam; }

/* Size of the maps for tables with 5 to 7 locked tiles, as a power of two (call before the first use). */
void nt_endgame_bits(int bits) { if (bits >= 16 && bits <= 34) eg_bits = bits; }
void nt_set_endgame(void *p, int kmin) { ((net_t *)p)->endgame = kmin < 0 ? 0 : kmin; }
int nt_get_endgame(void *p) { return ((net_t *)p)->endgame; }
/* 1 once any table ran out of room (a map at 80% or an allocation that failed): values may then be floors and play
 * may depend on timing. Evaluations should report it. */
int nt_endgame_full(void) { return eg_full; }
uint64_t nt_endgame_used(int k) { return k >= 0 && k < 16 ? eg_used[k] : 0; }
/* Family f's chance of winning for a board already turned onto its path, or -1 if it is not a position of its
 * table k. For tests. */
float nt_endgame_value_family(uint64_t lo, uint64_t hi, int k, int f) {
    board_t b = mk(lo, hi);
    uint8_t c[16];
    for (int i = 0; i < 16; i++) c[i] = (uint8_t)cell(b, i);
    if (k < EG_MIN || k > 15 || f < 0 || f >= eg_nfam || k > eg_kvalid[f] || (f && !eg_pre[f][k])) return -1.0f;
    for (int i = 0; i < k; i++) if (c[eg_path[f][i]] != 15 - i) return -1.0f;
    eg_caps_default();
    if (!eg_fits(f, c, k)) return -1.0f;
    pthread_mutex_lock(&eg_lock);
    float v = eg_value(f, c, k);
    pthread_mutex_unlock(&eg_lock);
    return v;
}
float nt_endgame_value(uint64_t lo, uint64_t hi, int k) { return nt_endgame_value_family(lo, hi, k, 0); }

/* The board the search evaluates for the root b: downgraded (see downgrade) while it holds the network's threshold
 * tile and the stage rule allows. */
static board_t root_board(const net_t *net, board_t b) {
    /* Again while the board still holds the threshold tile: after a 65536 one step only brings it to 32768, a
     * board the stages rarely saw; a second step shows it as the 16384 board the first 32768 was shown as. A board
     * whose largest tile is the threshold itself drops below it in one step, so play before a 65536 is unchanged. */
    for (int rep = 0; rep < 4 && net->downgrade && max_exp(b) >= net->downgrade; rep++) {
        /* Only when the downgraded board belongs to the same stage: a stage made for boards with a 32768
         * and a 16384 must see them as they are, or it would never be asked. */
        board_t lower = downgrade(b);
        if (lower == b) break;
        /* A tile above 32768 is beyond what any stage trained on: downgrade it whatever stage that gives. */
        if (!(max_exp(b) > 15 || stage_of(lower, net->bounds, net->n_bounds) == stage_of(b, net->bounds, net->n_bounds)))
            break;
        b = lower;
    }
    return b;
}

void nt_root_board(void *p, uint64_t lo, uint64_t hi, uint64_t *out_lo, uint64_t *out_hi) {
    board_t r = root_board((net_t *)p, mk(lo, hi));
    *out_lo = lo64(r);
    *out_hi = hi64(r);
}

static int best_move(search_ctx_t *c, board_t b, int depth) {
    if (c->net->endgame) {                                     /* an exact table, when the board is one of its positions */
        int d = eg_choose(b, c->net->endgame);
        if (d >= 0) return d;
    }
    board_t real = b;
    b = root_board(c->net, b);
    int cap = c->net->depth_cap[count_empty(b)];
    if (cap && depth > cap) depth = cap;
    int best = -1;
    double bestv = 0.0;
    for (int d = 0; d < 4; d++) {
        int r;
        board_t a = do_move(b, d, &r);
        if (a == b) continue;
        if (b != real && do_move(real, d, &r) == real) continue;   /* legal on the twin only (tiles at the top) */
        a = do_move(b, d, &r);
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
    long *next, n, first;   /* game g of the job is game first + g of the seed */
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
        uint64_t rng = game_seed(j->seed, j->first + g);
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

/* Games first .. first + games - 1 of the seed. A game depends on its number and
 * the seed alone, so an evaluation can be cut into pieces, run anywhere in any
 * order, and add up to the same games. */
/* One game of the seed, played exactly as nt_play_from plays it, with its history kept: the board before
 * each move (as two 64-bit halves) and the move made, then the final board. At most `cap` boards are kept
 * (the game is played to its end regardless); *n_boards is how many were. Returns the score. */
int64_t nt_play_trace(void *p, long game, int depth, double cutoff, uint64_t seed, long cap,
                      uint64_t *boards_lohi, int8_t *dirs, long *n_boards, int32_t *maxtile, int32_t *moves) {
    net_t *net = (net_t *)p;
    search_ctx_t *c = ctx_create(net, cutoff);
    uint64_t rng = game_seed(seed, game);
    board_t b = spawn(spawn(0, &rng), &rng);
    int64_t sc = 0;
    int32_t mv = 0;
    long kept = 0;
    for (;;) {
        int d = best_move(c, b, depth);
        if (kept < cap) {
            boards_lohi[2 * kept] = lo64(b);
            boards_lohi[2 * kept + 1] = hi64(b);
            dirs[kept] = (int8_t)d;                      /* -1 on the final board */
            kept++;
        }
        if (d < 0) break;
        int r;
        b = spawn(do_move(b, d, &r), &rng);
        sc += r;
        mv++;
    }
    ctx_free(c);
    *n_boards = kept;
    *maxtile = max_exp(b) ? (1 << max_exp(b)) : 0;
    *moves = mv;
    return sc;
}

void nt_play_from(void *p, long games, long first, int depth, double cutoff, int threads, uint64_t seed,
                  int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    long next = 0;
    play_job_t job = {(net_t *)p, depth, cutoff, &next, games, first, seed, scores, maxtiles, moves};
    run_workers(play_worker, &job, threads);
}

void nt_play(void *p, long games, int depth, double cutoff, int threads, uint64_t seed,
             int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    nt_play_from(p, games, 0, depth, cutoff, threads, seed, scores, maxtiles, moves);
}

/* Play games and keep, from each, the first board that holds a tile of exponent
 * `exp` or larger (after the spawn, as the player sees it): cells[16g..16g+15]
 * get its exponents and scores[g] the points so far. A game that ends first
 * leaves its cells at zero. */
typedef struct {
    net_t *net;
    int depth;
    int64_t scale;
    double cutoff;
    long *next, n;
    uint64_t seed;
    uint8_t *cells;
    int64_t *scores;
} collect_job_t;

static void *collect_worker(void *arg) {
    collect_job_t *j = (collect_job_t *)arg;
    search_ctx_t *c = ctx_create(j->net, j->cutoff);
    for (;;) {
        long g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->n) break;
        uint64_t rng = game_seed(j->seed, g);
        board_t b = spawn(spawn(0, &rng), &rng);
        int64_t sc = 0;
        while (board_scale(b) < j->scale) {
            int d = best_move(c, b, j->depth);
            if (d < 0) break;
            int r;
            b = spawn(do_move(b, d, &r), &rng);
            sc += r;
        }
        if (board_scale(b) >= j->scale)
            for (int i = 0; i < 16; i++) j->cells[16 * g + i] = (uint8_t)cell(b, i);
        j->scores[g] = sc;
    }
    ctx_free(c);
    return NULL;
}

/* `scale` is a threshold on the tiles a board holds, as for stage boundaries (see board_scale): 16384 keeps
 * the first board with a 16384 or larger, 49152 the first with a 32768 and a 16384, or larger. */
void nt_collect(void *p, long games, int depth, double cutoff, int threads, uint64_t seed, int64_t scale,
                uint8_t *cells, int64_t *scores) {
    long next = 0;
    memset(cells, 0, (size_t)games * 16);
    collect_job_t job = {(net_t *)p, depth, scale, cutoff, &next, games, seed, cells, scores};
    run_workers(collect_worker, &job, threads);
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
        size_t h = tt_hash(c, a);
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
    if (depth < 1) depth = 1;
    if (topk < 1) topk = 1;
    long next = 0;
    choose_job_t job = {(net_t *)p, depth, topk, &next, games, max_moves, prefix, seed, scores, maxtiles, moves};
    run_workers(choose_worker, &job, threads);
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
    int16_t fours;           /* 4s placed along the line */
} beam_t;

static inline int next_value(net_t *net, board_t s, double snake, double *out) {
    double best = 0.0;
    int any = 0;
    for (int m = 0; m < 4; m++) {
        int r;
        board_t a = do_move(s, m, &r);
        if (a == s) continue;
        double v = (double)r + net_value(net, a);
        if (snake > 0.0) v += snake * snake_score(net->snake_w, a);
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
/* tiles: bitmask of placements allowed, 1 = a 2, 2 = a 4 (3 = both). Placing only 2s
 * earns every 2+2 merge and approaches the theoretical maximum score, but the
 * 131072 needs one 4 at the very end; bit 4 charges each placed 4 the 4 points it
 * forfeits (in the ranking only). Bits 8 and 16 are handled by beam_search below.
 * max_fours: a line may place at most this many 4s (negative = no limit).
 * *alive gets whether the best line can still move at the end of the horizon. */
static int beam_search_raw(net_t *net, board_t b, int width, int depth, int spread, double snake, int tiles,
                           int max_fours, int *moves, int *cells, int *values, double *score, int *alive) {
    if (width < 1) width = 1;
    if (depth < 1) depth = 1;
    if (spread < 0) spread = 0;
    if ((tiles & 3) == 0) tiles = 3;
    size_t cap = (size_t)width * 121;            /* 4 moves x 30 placements, plus carried dead lines */
    size_t hcap = 1;
    while (hcap < 2 * cap) hcap <<= 1;
    int hbits = 0;
    while (((size_t)1 << hbits) < hcap) hbits++;
    beam_t *lev = (beam_t *)malloc(sizeof(beam_t) * (size_t)width * (size_t)(depth + 1));
    beam_t *cand = (beam_t *)malloc(sizeof(beam_t) * cap);
    board_t *seen = (board_t *)malloc(sizeof(board_t) * hcap);
    int *kids = (int *)malloc(sizeof(int) * (size_t)width);
    lev[0] = (beam_t){b, 0.0, 0, -1, 1, -1, -1, 0, 0};
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
                        if (!(tiles & (1 << (v - 1)))) continue;
                        if (v == 2 && max_fours >= 0 && e->fours >= max_fours) continue;
                        board_t s2 = a | put((int)v, c);
                        size_t h = (size_t)(hash128(s2) >> (64 - hbits));
                        while (seen[h] && seen[h] != s2) h = (h + 1) & (hcap - 1);
                        if (seen[h] == s2) continue;
                        seen[h] = s2;
                        beam_t ne;
                        ne.s = s2;
                        ne.cum = e->cum + r - ((tiles & 4) && v == 2 ? 4 : 0);
                        ne.parent = i;
                        double nv;
                        ne.alive = (int8_t)next_value(net, s2, snake, &nv);
                        ne.score = (double)ne.cum + (ne.alive ? nv : 0.0);
                        ne.m = (int8_t)m;
                        ne.cell = (int8_t)c;
                        ne.val = (int8_t)v;
                        ne.fours = (int16_t)(e->fours + (v == 2));
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
        *alive = lev[(size_t)L * width].alive;
    } else {
        *score = 0.0;
        *alive = 0;
    }
    free(lev);
    free(cand);
    free(seen);
    free(kids);
    return len;
}

/* Bit 8 of tiles = 2s first: plan with 2s only, and only when the best 2s-only
 * line dies within the horizon plan again with 4s allowed (bit 4 still charges
 * them). With a snake bonus and no negative values a live line always outranks a
 * dead one, so that means every 2s-only line in the beam dies. A fallback plan is
 * cut to its first step, so the next plan tries 2s again: 4s are placed only when
 * 2s cannot keep the game going.
 *
 * Bit 16 = as few 4s as possible: the fallback first looks for a line that
 * survives the horizon with a single 4, then two, and so on, before it allows any
 * number. A free fallback takes a 4 at every step it is asked, seven or eight per
 * cascade, where one is what the cascade needs. And a tile that ends the game
 * whatever it is, the last cell of a full board, is placed as a 4: the score is
 * the same, and the final board is the one the score formula is built on. */
static int beam_search(net_t *net, board_t b, int width, int depth, int spread, double snake, int tiles,
                       int *moves, int *cells, int *values, double *score) {
    int alive;
    if (!(tiles & 8))
        return beam_search_raw(net, b, width, depth, spread, snake, tiles, -1, moves, cells, values, score, &alive);
    int len = beam_search_raw(net, b, width, depth, spread, snake, 1, -1, moves, cells, values, score, &alive);
    if (len > 0 && alive) return len;
    if (tiles & 16) {
        for (int k = 1; k <= depth; k++) {
            len = beam_search_raw(net, b, width, depth, spread, snake, 3 | (tiles & 4), k, moves, cells, values, score, &alive);
            if (len > 0 && alive) return 1;
        }
    }
    len = beam_search_raw(net, b, width, depth, spread, snake, 3 | (tiles & 4), -1, moves, cells, values, score, &alive);
    if (len > 0 && (tiles & 16) && values[0] == 1) {
        int r;
        board_t a = do_move(b, moves[0], &r);
        double nv;
        if (count_empty(a) == 1 && !next_value(net, a | put(1, cells[0]), snake, &nv) &&
            !next_value(net, a | put(2, cells[0]), snake, &nv))
            values[0] = 2;
    }
    return len < 1 ? len : 1;
}

#define BEAM_MAX_DEPTH 256

int nt_beam_plan(void *p, uint64_t lo, uint64_t hi, int width, int depth, int spread, double snake, int tiles,
                 int *moves, int *cells, int *values) {
    board_t b = mk(lo, hi);
    double score;
    if (depth > BEAM_MAX_DEPTH) depth = BEAM_MAX_DEPTH;
    return beam_search((net_t *)p, b, width, depth, spread, snake, tiles, moves, cells, values, &score);
}

int nt_beam_choose(void *p, uint64_t lo, uint64_t hi, int width, int depth, int spread, double snake, int tiles,
                   int *cell_out, int *value) {
    board_t b = mk(lo, hi);
    int moves[BEAM_MAX_DEPTH], cells[BEAM_MAX_DEPTH], values[BEAM_MAX_DEPTH];
    double score;
    if (depth > BEAM_MAX_DEPTH) depth = BEAM_MAX_DEPTH;
    int len = beam_search((net_t *)p, b, width, depth, spread, snake, tiles, moves, cells, values, &score);
    if (len == 0) { *cell_out = -1; *value = 0; return -1; }
    *cell_out = cells[0];
    *value = values[0];
    return moves[0];
}

double nt_beam_value(void *p, uint64_t lo, uint64_t hi, int width, int depth, int spread, double snake, int tiles) {
    board_t b = mk(lo, hi);
    int moves[BEAM_MAX_DEPTH], cells[BEAM_MAX_DEPTH], values[BEAM_MAX_DEPTH];
    double score;
    if (depth > BEAM_MAX_DEPTH) depth = BEAM_MAX_DEPTH;
    beam_search((net_t *)p, b, width, depth, spread, snake, tiles, moves, cells, values, &score);
    return score;
}

typedef struct {
    net_t *net;
    int width, depth, stride, spread, tiles;
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
            int len = beam_search(j->net, b, j->width, j->depth, j->spread, j->snake, j->tiles, moves, cells, values, &score);
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

void nt_play_beam(void *p, long games, int width, int depth, int stride, int spread, double snake, int tiles,
                  int threads, uint64_t seed, long max_moves, long prefix, int64_t *scores, int32_t *maxtiles,
                  int32_t *moves) {
    if (stride < 1) stride = 1;
    if (depth > BEAM_MAX_DEPTH) depth = BEAM_MAX_DEPTH;
    long next = 0;
    beam_job_t job = {(net_t *)p, width, depth, stride, spread, tiles, snake, &next, games, max_moves, prefix, seed,
                      scores, maxtiles, moves};
    run_workers(beam_worker, &job, threads);
}

/* ------------------------------------------------ learning the choose game --- */
/* TD(0) on afterstates, mirroring learn_game: the policy is the choose search at
 * `depth` (1 = every placement, then every reply move) using the current weights,
 * so the value function learns the game where the player places the tiles. */

static void learn_choose_game(net_t *net, search_ctx_t *c, int depth, int topk, long max_moves, double explore,
                              const board_t *starts, long n_starts, double start_frac,
                              float alpha, float alpha_plain, uint64_t *rng,
                              int64_t *score, int32_t *maxtile, int32_t *nmoves) {
    /* late-game restarts: a fraction of games begin from a saved board, so the
     * endgame gets more than one game-over per 30k moves of training signal */
    board_t b;
    if (n_starts > 0 && start_frac > 0.0 && (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0) < start_frac)
        b = starts[rng_next(rng) % (uint64_t)n_starts];
    else
        b = spawn(spawn(0, rng), rng);
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
    const board_t *starts;
    long n_starts;
    double start_frac;
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
        learn_choose_game(j->net, c, j->depth, j->topk, j->max_moves, j->explore, j->starts, j->n_starts, j->start_frac,
                          j->alpha, j->alpha_plain, &rng, &j->scores[g], &j->maxtiles[g], &j->moves[g]);
    }
    ctx_free(c);
    return NULL;
}

/* starts_lohi: n_starts boards as (lo, hi) pairs; start_frac of the games begin from one. */
void nt_train_choose(void *p, long games, int threads, float alpha, float alpha_plain, uint64_t seed,
                     int depth, int topk, long max_moves, double explore,
                     const uint64_t *starts_lohi, long n_starts, double start_frac,
                     int64_t *scores, int32_t *maxtiles, int32_t *moves) {
    net_t *net = (net_t *)p;
    if (depth < 1) depth = 1;
    if (topk < 1) topk = 1;
    board_t *starts = unpack_boards(starts_lohi, &n_starts);
    long next = 0;
    choose_train_job_t job = {net, depth, topk, &next, games, max_moves, explore, starts, n_starts, start_frac,
                              alpha, alpha_plain, seed, scores, maxtiles, moves};
    run_workers(choose_train_worker, &job, threads);
    free(starts);
    net->version++;
}

/* ------------------------------------------------------------- persistence --- */

static const char MAGIC_V1[8] = "NTUPLE01";
static const char MAGIC_V2[8] = "NTUPLE02";
static const char MAGIC_V3[8] = "NTUPLE03";
static const char MAGIC_V4[8] = "NTUPLE04";
static const char MAGIC_V5[8] = "NTUPLE05";
static const char MAGIC_V6[8] = "NTUPLE06";   /* radix-18 tables (five-bit cells) */

/* One quantity of a table as the plain array the file format holds: float `off` of every record. */
static void write_column(FILE *f, const float *t, int sh, int off, size_t n, float *buf) {
    if (sh == 0) { fwrite(t, sizeof(float), n, f); return; }
    for (size_t i = 0; i < n; i++) buf[i] = t[(i << sh) + off];
    fwrite(buf, sizeof(float), n, f);
}

/* Format 06: header (K, tc_stages, flags, n_bounds, bounds, len, orig), then per
 * stage and pattern: w; e and a for TC stages; touched bits for plain stages >= 1.
 * weights_only drops the learning state and writes every stage's table with
 * promotion applied, so the file plays identically at a third of the size.
 * Returns 0, or -1 when the file could not be written in full. */
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
    if ((weights_only && net->flags) || net->tc_stages > 0) {   /* scratch: baking promotion in, or one column of a record table */
        size_t largest = 0;
        for (int k = 0; k < net->K; k++) if (net->size[k] > largest) largest = net->size[k];
        buf = (float *)malloc(largest * sizeof(float));
        if (!buf) { fclose(f); return -1; }
    }
    for (int st = 0; st < net->n_stages; st++) {
        for (int k = 0; k < net->K; k++) {
            if (weights_only && st > 0 && net->flags) {
                for (uint32_t i = 0; i < net->size[k]; i++) buf[i] = entry_value(net, st, k, i);
                fwrite(buf, sizeof(float), net->size[k], f);
            } else {
                write_column(f, net->w[st][k], stage_shift(net, st), 0, net->size[k], buf);
            }
            if (!weights_only && stage_is_tc(net, st)) {
                write_column(f, net->w[st][k], 2, 1, net->size[k], buf);
                write_column(f, net->w[st][k], 2, 2, net->size[k], buf);
            }
            if (!weights_only && stage_has_bits(net, st)) fwrite(net->bits[st][k], 1, bits_bytes(net->size[k]), f);
        }
    }
    free(buf);
    int failed = ferror(f);                          /* e.g. the disk filled up */
    return (fclose(f) == 0 && !failed) ? 0 : -1;
}

static int skip_bytes(FILE *f, size_t n) { return fseek(f, (long)n, SEEK_CUR) == 0 ? 0 : -1; }

/* Formats 1-5 indexed tables with 4 bits per cell; entry `old` of an n-cell table
 * moves to this radix-18 index (entries with a digit above 15 stay zero). */
static inline uint32_t remap_old_index(uint32_t old, int n) {
    uint32_t idx = 0;
    for (int j = 0; j < n; j++) idx = idx * RADIX + ((old >> (4 * (n - 1 - j))) & 0xF);
    return idx;
}

/* Read a table of `fn` floats (NULL = skip) into float `off` of each record of dst, records of 1 << sh
 * floats; re-index when the file is old. */
static int read_table(FILE *f, float *dst, int sh, int off, size_t fn, int old, int n, float *tmp) {
    if (!dst) return skip_bytes(f, fn * sizeof(float));
    if (!old && sh == 0) return fread(dst, sizeof(float), fn, f) == fn ? 0 : -1;
    if (fread(tmp, sizeof(float), fn, f) != fn) return -1;
    for (uint32_t i = 0; i < fn; i++) dst[((size_t)(old ? remap_old_index(i, n) : i) << sh) + off] = tmp[i];
    return 0;
}

static int read_bits(FILE *f, uint8_t *dst, size_t fn, int old, int n, uint8_t *tmpb) {
    size_t nb = bits_bytes(fn);
    if (!dst) return skip_bytes(f, nb);
    if (!old) return fread(dst, 1, nb, f) == nb ? 0 : -1;
    if (fread(tmpb, 1, nb, f) != nb) return -1;
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
        f_a_all = has_a < 0 ? use_tc : has_a;     /* formats 1-2 stored A exactly when they stored E */
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
    uint8_t *tmpb = (uint8_t *)malloc(bits_bytes(maxfn));
    if (!tmp || !tmpb) goto fail;
    for (int st = 0; st < file_stages; st++) {
        for (int k = 0; k < K; k++) {
            size_t fn = old ? ((size_t)1 << (4 * len[k])) : net->size[k];
            int f_has_e = v >= 5 ? st < f_tc_stages : f_e_all;
            int f_has_a = v >= 5 ? st < f_tc_stages : f_a_all;
            int f_has_bits = v >= 5 ? (f_flags && st > 0 && st >= f_tc_stages) : (f_bits_v4 && st > 0);
            if (read_table(f, net->w[st][k], stage_shift(net, st), 0, fn, old, len[k], tmp)) goto fail;
            if (f_has_e && read_table(f, stage_is_tc(net, st) ? net->w[st][k] : NULL, 2, 1, fn, old, len[k], tmp)) goto fail;
            if (f_has_a) {
                if (stage_is_tc(net, st)) { if (read_table(f, net->w[st][k], 2, 2, fn, old, len[k], tmp)) goto fail; }
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
