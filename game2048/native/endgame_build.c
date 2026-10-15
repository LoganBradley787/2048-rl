/* Endgame tables built whole in advance, for the engine's --endgame play (nt_endgame_load in ntuple.c).
 * Run it through examples/build_endgame.py.
 *
 * The game is the one the engine solves on demand: the k largest tiles, 32768 down, locked one per cell along
 * the snake; value = chance of merging a tile into the last locked one with the locked tiles never moving, or of
 * locking the next rung when it is in place and going on in table k+1. Tables are filled from k = 15 down.
 * Caps keep the large tables storable: on tables with at most kcap locked tiles, the tile on the j-th free snake
 * cell is at most 2^(n - d_j) (n = 16 - k free cells; the last d repeats; never below 4, never above n).
 *
 * Layout: positions sorted by the sum of the tiles in play (in units of 2), then by rank within the sum, the
 * first free cell most significant. offs[s] counts the positions of smaller sum; the rank adds, cell by cell, the
 * number of ways to fill the later cells with what is left when this cell holds a smaller tile (PR[k][j]).
 * A move keeps the sum and a new tile adds 1 or 2 units, so sum s reads only sums s+1 and s+2 of its own table
 * (stored just after it) and table k+1: the files are written in place through a shared mapping, and no table
 * needs to fit in memory. Values: uint16, v * 65535 rounded.
 *
 * Program:  endgame_build <kmin> <d0,d1,...> <outdir> [threads] [kcap] [path|-] [lift] [wobble]
 *           writes <outdir>/k<k>.u16 for k = kmin .. 15 and <outdir>/caps.txt
 *           ("kmin drops kcap 2 [path [lift [wobble]]]"; wobble: 1 = the block may slide, see WOBBLE).
 *           path: 16 cells, comma-separated, the order the big tiles lie in (default: the snake
 *           0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12); consecutive locked cells must be neighbours.
 * Library:  tb2_build, tb2_load, tb2_value, tb2_size (for checks).
 */
#include <fcntl.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int SNAKE[16] = {0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12};   /* the formation's cell path */
static int PATHSET = 0;
/* tables are built up to KVALID locked tiles: the longest prefix of the path whose consecutive cells are neighbours
 * (beyond it a merge into the last locked tile would not collapse the ladder); tables above it are left at 0 */
static int KVALID = 15;
/* LIFT g: sub-goal tables. The i-th locked tile is 2^(15 - i - g), so the last one (the tile to merge into) is
 * 2^(16 - k - g) and tiles in play go up to it. In a real game the tiles above a gap in the ladder are boulders
 * (see eg_orient in ntuple.c): with 32768 16384 | 4096 2048 1024 the 8192 is being built, a table with g = 1. */
static int LIFT = 0;
static inline int top_of(int k) { return 16 - k - LIFT; }
static int chain_length(void) {
    int k = 1;
    while (k < 16) {
        int a = SNAKE[k - 1], b = SNAKE[k], dr = a / 4 - b / 4, dc = a % 4 - b % 4;
        if (dr * dr + dc * dc != 1) break;
        k++;
    }
    return k > 15 ? 15 : k;
}
/* WOBBLE: the locked tiles after the last full row of the path (the "block") may slide along their row as one
 * piece; the rows above stay locked. A table position is then (offset, tiles in play): the block sits on path
 * cells rowstart + o .. rowstart + o + L - 1, and the free cells are the other path cells from rowstart on, in path
 * order (offset 0: the cells after the block, as without wobble). A merge into the last locked tile wins only with
 * the block at home (o = 0). Of several readings of a board the smallest offset is used. Rows of the path are its
 * cells 4r .. 4r + 3 (the snake's; other paths are refused). File layout: offset-major, each offset laid out as a
 * table without wobble, so offset 0 is where the plain table's values would be. */
static int WOBBLE = 0;
static int NOFF[16], FREE[16][4][16];       /* FREE[k][o][j]: path index of free cell j of table k at offset o */
static inline int rowstart(int k) { return ((k - 1) / 4) * 4; }
static inline int blen(int k) { return k - rowstart(k); }
static int DROP[16], NDROP, KCAP = 15;
static int CAP[16][16], MAXS[16];
static uint64_t *OFFS[16], *PR[16][16], SIZE[16];
static uint16_t *V[16];

static inline int units(int e) { return e ? 1 << (e - 1) : 0; }

static int cap_of(int k, int j) {
    int n = top_of(k), d = NDROP && k <= KCAP ? DROP[j < NDROP ? j : NDROP - 1] : 0, c = n - d;
    if (c < 2) c = 2;
    return c > n ? n : c;
}

static void setup(int k) {
    int n = 16 - k;
    NOFF[k] = WOBBLE ? 4 - blen(k) + 1 : 1;
    for (int o = 0; o < NOFF[k]; o++) {
        int j = 0;
        for (int q = WOBBLE ? rowstart(k) : k; q < 16; q++)
            if (q < rowstart(k) + o || q >= rowstart(k) + o + blen(k)) FREE[k][o][j++] = q;
    }
    for (int j = 0; j < n; j++) CAP[k][j] = cap_of(k, j);
    int S = 0;
    for (int j = 0; j < n; j++) S += units(CAP[k][j]);
    MAXS[k] = S;
    /* cnt[j][r]: ways to fill cells j .. n-1 with r units */
    uint64_t *cnt = calloc((size_t)(n + 1) * (S + 1), sizeof(uint64_t));
    cnt[(size_t)n * (S + 1)] = 1;
    for (int j = n - 1; j >= 0; j--)
        for (int r = 0; r <= S; r++) {
            uint64_t t = 0;
            for (int e = 0; e <= CAP[k][j] && units(e) <= r; e++) t += cnt[(size_t)(j + 1) * (S + 1) + r - units(e)];
            cnt[(size_t)j * (S + 1) + r] = t;
        }
    free(OFFS[k]); OFFS[k] = malloc((size_t)(S + 2) * sizeof(uint64_t));
    OFFS[k][0] = 0;
    for (int s = 0; s <= S; s++) OFFS[k][s + 1] = OFFS[k][s] + cnt[s];
    SIZE[k] = OFFS[k][S + 1];
    for (int j = 0; j < n; j++) {
        int C = CAP[k][j] + 1;
        free(PR[k][j]); PR[k][j] = malloc((size_t)(S + 1) * C * sizeof(uint64_t));
        for (int r = 0; r <= S; r++) {
            uint64_t acc = 0;
            for (int v = 0; v < C; v++) {
                PR[k][j][(size_t)r * C + v] = acc;
                if (units(v) <= r) acc += cnt[(size_t)(j + 1) * (S + 1) + r - units(v)];
            }
        }
    }
    free(cnt);
}

static inline int fits(const uint8_t *b, int k, int o) {
    for (int j = 0; j < 16 - k; j++) if (b[SNAKE[FREE[k][o][j]]] > CAP[k][j]) return 0;
    return 1;
}

static inline uint64_t index_of(const uint8_t *b, int k, int o) {
    int n = 16 - k, s = 0;
    for (int j = 0; j < n; j++) s += units(b[SNAKE[FREE[k][o][j]]]);
    uint64_t idx = (uint64_t)o * SIZE[k] + OFFS[k][s];
    for (int j = 0; j < n; j++) {
        int v = b[SNAKE[FREE[k][o][j]]];
        idx += PR[k][j][(size_t)s * (CAP[k][j] + 1) + v];
        s -= units(v);
    }
    return idx;
}

static inline float get(int k, int o, const uint8_t *b) { return (float)V[k][index_of(b, k, o)] * (1.0f / 65535.0f); }

/* the block of table k on board a (rows above already checked): the smallest offset whose block holds the locked
 * tiles with the last one either itself (*won = 0) or merged into (*won = 1); -1 if none */
static inline int find_block(const uint8_t *a, int k, int *won) {
    int rs = rowstart(k), L = blen(k), n = top_of(k);
    *won = 0;
    for (int o = 0; o < NOFF[k]; o++) {
        int m = 1;
        for (int i = 0; i < L - 1 && m; i++) if (a[SNAKE[rs + o + i]] != 15 - (rs + i) - LIFT) m = 0;
        if (!m) continue;
        int last = a[SNAKE[rs + o + L - 1]];
        if (last == n) return o;
        if (last == n + 1) { *won = 1; return o; }
    }
    return -1;
}

static int slide(uint8_t *l) {
    uint8_t t[4]; int n = 0, changed = 0;
    for (int i = 0; i < 4; i++) if (l[i]) t[n++] = l[i];
    uint8_t o[4] = {0, 0, 0, 0}; int m = 0;
    for (int i = 0; i < n;) {
        if (i + 1 < n && t[i] == t[i + 1]) { o[m++] = t[i] + 1; i += 2; }
        else o[m++] = t[i++];
    }
    for (int i = 0; i < 4; i++) { if (l[i] != o[i]) changed = 1; l[i] = o[i]; }
    return changed;
}

static int move(const uint8_t *b, int dir, uint8_t *out) {
    int changed = 0;
    memcpy(out, b, 16);
    for (int i = 0; i < 4; i++) {
        uint8_t l[4]; int c[4];
        for (int j = 0; j < 4; j++)
            c[j] = dir == 0 ? 4 * j + i : dir == 2 ? 4 * (3 - j) + i : dir == 3 ? 4 * i + j : 4 * i + (3 - j);
        for (int j = 0; j < 4; j++) l[j] = b[c[j]];
        changed |= slide(l);
        for (int j = 0; j < 4; j++) out[c[j]] = l[j];
    }
    return changed;
}

/* the value of position b of table k: the best allowed move, or locking the next rung (as eg_value in ntuple.c) */
static float solve(const uint8_t *b, int k, int o, int *best_dir) {
    int n = top_of(k), rs = rowstart(k), L = blen(k);   /* the last locked tile; tiles in play up to it */
    int above = WOBBLE ? rs : k - 1;                     /* tiles that never move (without wobble: all but the last) */
    float best = 0.0f; int bd = -1;
    for (int d = 0; d < 4; d++) {
        uint8_t a[16];
        if (!move(b, d, a)) continue;
        int ok = 1;
        for (int i = 0; i < above && ok; i++) if (a[SNAKE[i]] != 15 - i - LIFT) ok = 0;
        if (!ok) continue;
        int won, o2;
        if (WOBBLE) o2 = find_block(a, k, &won);
        else { o2 = 0; won = a[SNAKE[k - 1]] == n + 1; if (!won && a[SNAKE[k - 1]] != n) o2 = -1; }
        if (o2 < 0) continue;
        if (won) {
            if (o2 == 0) { best = 1.0f; bd = d; break; }
            continue;                                    /* merged away from home: not counted (a lower bound) */
        }
        if (!fits(a, k, o2)) continue;
        double sum = 0.0; int empty = 0;
        for (int c = 0; c < 16; c++) {
            if (a[c]) continue;
            empty++;
            a[c] = 1; sum += 0.9 * get(k, o2, a);
            a[c] = 2; sum += (n >= 2) ? 0.1 * get(k, o2, a) : 0.0;      /* a 4 is too large to be in play: lost */
            a[c] = 0;
        }
        float v = empty ? (float)(sum / empty) : 0.0f;
        if (bd < 0 || v > best) { best = v; bd = d; }
    }
    if (best_dir) *best_dir = bd;
    /* lock the next rung: the cell after the block holds it (in the same row the offset stays; a full row's
     * block is at home and the next rung starts a new row) */
    int q = rs + o + L;
    if (k < KVALID && (L < 4 ? q < rs + 4 : 1) && b[SNAKE[q]] == n - 1 && fits(b, k + 1, L < 4 ? o : 0)) {
        float w = get(k + 1, L < 4 ? o : 0, b);
        if (w > best) best = w;
    }
    return best;
}

static void fill_rest(int k, int o, uint8_t *b, int j, int rem, const int *maxrem) {
    int n = 16 - k;
    if (j == n) {
        if (rem) return;
        V[k][index_of(b, k, o)] = (uint16_t)(solve(b, k, o, NULL) * 65535.0f + 0.5f);
        return;
    }
    if (rem > maxrem[j]) return;
    int c = SNAKE[FREE[k][o][j]];
    for (int e = 0; e <= CAP[k][j] && units(e) <= rem; e++) {
        b[c] = (uint8_t)e;
        fill_rest(k, o, b, j + 1, rem - units(e), maxrem);
    }
    b[c] = 0;
}

static void build_table(int k, int threads) {
    int n = 16 - k;
    int maxrem[17]; maxrem[n] = 0;
    for (int j = n - 1; j >= 0; j--) maxrem[j] = maxrem[j + 1] + units(CAP[k][j]);
    int split = n >= 2 ? 2 : 1;
    int c0 = CAP[k][0] + 1, c1 = split == 2 ? CAP[k][1] + 1 : 1, tasks = c0 * c1 * NOFF[k];
    int rs = rowstart(k), L = blen(k);
    time_t t0 = time(NULL);
    for (int s = maxrem[0]; s >= 0; s--) {
        #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (int t = 0; t < tasks; t++) {
            int e0 = t % c0, e1 = (t / c0) % c1, o = t / (c0 * c1);
            int rem = s - units(e0) - (split == 2 ? units(e1) : 0);
            if (rem < 0 || rem > maxrem[split]) continue;
            uint8_t b[16];
            memset(b, 0, sizeof b);
            if (WOBBLE) {
                for (int i = 0; i < rs; i++) b[SNAKE[i]] = (uint8_t)(15 - i - LIFT);
                for (int i = 0; i < L; i++) b[SNAKE[rs + o + i]] = (uint8_t)(15 - (rs + i) - LIFT);
            } else
                for (int i = 0; i < k; i++) b[SNAKE[i]] = (uint8_t)(15 - i - LIFT);
            b[SNAKE[FREE[k][o][0]]] = (uint8_t)e0;
            if (split == 2) b[SNAKE[FREE[k][o][1]]] = (uint8_t)e1;
            fill_rest(k, o, b, split, rem, maxrem);
        }
        if (k <= 6 && s % 64 == 0)
            fprintf(stderr, "  k %d sum %d done (%.1f%% of positions), %lds\n", k, 2 * s,
                    100.0 * (double)(SIZE[k] - OFFS[k][s]) / (double)SIZE[k], (long)(time(NULL) - t0));
    }
}

static void set_drops(const int *drops, int ndrops, int kcap) {
    NDROP = ndrops < 16 ? ndrops : 16;
    for (int j = 0; j < NDROP; j++) DROP[j] = drops[j];
    KCAP = kcap;
}

static uint16_t *map_file(const char *path, uint64_t n, int write) {
    int fd = open(path, write ? O_RDWR | O_CREAT | O_TRUNC : O_RDONLY, 0644);
    if (fd < 0) return NULL;
    if (write && ftruncate(fd, (off_t)(n * 2)) != 0) { close(fd); return NULL; }
    if (!write && lseek(fd, 0, SEEK_END) != (off_t)(n * 2)) { close(fd); return NULL; }
    void *m = mmap(NULL, n * 2, write ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return m == MAP_FAILED ? NULL : (uint16_t *)m;
}

int tb2_build(int kmin, const int *drops, int ndrops, int kcap, const char *dir, int threads) {
    set_drops(drops, ndrops, kcap);
    KVALID = chain_length();
    if (KVALID > 15 - LIFT) KVALID = 15 - LIFT;         /* the last locked tile must be at least a 2 */
    char path[4096];
    snprintf(path, sizeof path, "%s/caps.txt", dir);
    unlink(path);                                     /* written last: a directory without it is not a finished build */
    for (int k = 15 - LIFT; k >= kmin; k--) {
        setup(k);
        snprintf(path, sizeof path, "%s/k%d.u16", dir, k);
        if (!(V[k] = map_file(path, SIZE[k] * NOFF[k], 1))) return -1;
        time_t t = time(NULL);
        if (k <= KVALID) build_table(k, threads);    /* else the file stays all zeros */
        msync(V[k], SIZE[k] * NOFF[k] * 2, MS_ASYNC);
        fprintf(stderr, "table %d: %llu positions, %lds\n", k, (unsigned long long)(SIZE[k] * NOFF[k]),
                (long)(time(NULL) - t));
    }
    for (int k = kmin; k <= 15 - LIFT; k++) msync(V[k], SIZE[k] * NOFF[k] * 2, MS_SYNC);
    snprintf(path, sizeof path, "%s/caps.txt", dir);
    FILE *f = fopen(path, "w"); if (!f) return -1;
    fprintf(f, "%d ", kmin);
    for (int j = 0; j < NDROP; j++) fprintf(f, j ? ",%d" : "%d", DROP[j]);
    fprintf(f, " %d 2", KCAP);
    if (PATHSET || LIFT || WOBBLE) for (int i = 0; i < 16; i++) fprintf(f, i ? ",%d" : " %d", SNAKE[i]);
    if (LIFT || WOBBLE) fprintf(f, " %d", LIFT);
    if (WOBBLE) fprintf(f, " wobble");
    fprintf(f, "\n");
    return fclose(f) == 0 ? 0 : -1;
}

int tb2_load(const char *dir, int kmin, const int *drops, int ndrops, int kcap) {
    set_drops(drops, ndrops, kcap);
    char path[4096];
    for (int k = kmin; k <= 15 - LIFT; k++) {
        setup(k);
        snprintf(path, sizeof path, "%s/k%d.u16", dir, k);
        if (!(V[k] = map_file(path, SIZE[k] * NOFF[k], 0))) return -1;
    }
    return 0;
}

unsigned long long tb2_size(int k) { return (unsigned long long)SIZE[k]; }

/* board: 16 exponents turned so the locked tiles are on the snake; -1 if not a position of table k */
float tb2_value(const uint8_t *b, int k) {
    if (k < 1 || k > 15 || !V[k]) return -1.0f;
    int above = WOBBLE ? rowstart(k) : k, won, o = 0;
    for (int i = 0; i < above; i++) if (b[SNAKE[i]] != 15 - i - LIFT) return -1.0f;
    if (WOBBLE && ((o = find_block(b, k, &won)) < 0 || won)) return -1.0f;
    if (!fits(b, k, o)) return -1.0f;
    return get(k, o, b);
}

void tb2_set_wobble(int w) { WOBBLE = w; }

#ifdef EGB_MAIN
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "endgame_build <kmin> <d0,d1,...> <outdir> [threads] [kcap] [path|-] [lift] [wobble]\n"); return 2; }
    int kmin = atoi(argv[1]), drops[16], nd = 0, threads = argc > 4 ? atoi(argv[4]) : 1;
    int kcap = argc > 5 ? atoi(argv[5]) : 15;
    if (argc > 7) LIFT = atoi(argv[7]);
    if (argc > 8) WOBBLE = atoi(argv[8]) != 0;
    if (LIFT < 0 || LIFT > 8) { fprintf(stderr, "lift must be 0 .. 8\n"); return 2; }
    if (argc > 6 && strcmp(argv[6], "-") != 0) {
        int n = 0; unsigned seen = 0;
        for (char *p = argv[6]; *p && n < 17;) {
            int c = atoi(p);
            if (c < 0 || c > 15 || n == 16) { n = 17; break; }
            SNAKE[n++] = c; seen |= 1u << c;
            while (*p && *p != ',') p++;
            if (*p) p++;
        }
        if (n != 16 || seen != 0xFFFF) { fprintf(stderr, "path must list the 16 cells once each\n"); return 2; }
        PATHSET = 1;
    }
    if (WOBBLE && PATHSET) {
        for (int i = 0; i < 16; i++)
            if (SNAKE[i] / 4 != i / 4 || (i % 4 && abs(SNAKE[i] - SNAKE[i - 1]) != 1)) { fprintf(stderr, "wobble needs a path that runs row by row\n"); return 2; }
    }
    for (char *p = argv[2]; *p && nd < 16;) { drops[nd++] = atoi(p); while (*p && *p != ',') p++; if (*p) p++; }
    if (tb2_build(kmin, drops, nd, kcap, argv[3], threads)) { fprintf(stderr, "build failed (memory or files)\n"); return 1; }
    return 0;
}
#endif
