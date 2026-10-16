/* Block-tier tables, built whole in advance (the engine's block tier: nt_block_load in ntuple.c). Run it through
 * examples/build_endgame.py --block. Ideas from macroxue/2048-ai's Tuple10 / Tuple11 (see
 * notes/design/macroxue_port.md); the code is ours.
 *
 * Shapes (cells row-major, row 0 at the top; the engine tries all 8 orientations of the board):
 *   b6: the 2x3 block r0c0 r0c1 r0c2 / r1c0 r1c1 r1c2 is locked; anchor a = min(r1c0, r1c1, r1c2).
 *       Regular: r0c0, r0c1 not empty; neighbours inside the block differ (r0c0 r0c1, r0c1 r0c2, r1c0 r1c1,
 *       r1c1 r1c2, r0c0 r1c0, r0c1 r1c1) and r0c2 > r1c2.
 *       Free cells and caps: r0c3, r2c0 <= a-1; r1c3, r2c1 <= a-2; r2c2, r2c3, row 3 <= a-3.
 *       Goal (a set-up of a tile of rank a beside the block, three merges away): r2c0 = a-1, r2c1 = a-2 and
 *       (r2c2 = a-3 with another a-3 at r3c2 or r2c3, or r3c1 = a-3 with another at r3c0 or r3c2); or
 *       r0c3 = a-1, r1c3 = a-2, r2c3 = a-3 with another a-3 at r2c2 or r3c3.
 *   b5: r0c0 r0c1 r0c2 / r1c0 r1c1 locked; anchor a = min(r1c1, r0c2); the goal tile is t = a-1 at r1c2.
 *       Regular: r0c0, r0c1 not empty; r0c0 != r0c1, r0c1 != r0c2, r0c0 != r1c0, r0c1 != r1c1, r1c1 != r0c2,
 *       r1c0 > r1c1. Free cells and caps: r1c2 <= a-1; r0c3, r1c3, row 2 <= a-2; row 3 <= a-3.
 *       Goal (one merge from a twin of the anchor): r1c2 = t and (r1c3 = t-1 with another t-1 at r0c3 or r2c3;
 *       or column 3 = t-1, t-2, t-2 from the top; or r2c2 = t-1 with another t-1 at r2c1, r3c2 or r2c3).
 * Under these conditions no free tile can merge with a block tile (every free cell is below each block
 * neighbour) and the block tiles never merge with each other, so the game played on the free cells does not
 * depend on the block's values: a table is indexed by the free cells only and serves every block with anchor
 * >= a (the engine uses the largest table a <= the board's anchor). The block's cells are walls here.
 *
 * Game: a move is allowed when it changes the board, leaves every block cell as it was (a block row or column
 * slides when a free cell in its way is empty: not allowed) and keeps every free tile under its cap; it wins
 * (value 1) when its afterstate is a goal. Otherwise the value is the average over the new tile (2 with 0.9,
 * 4 with 0.1, on every empty cell, equally likely) of the best move's value; a new tile above its cell's cap
 * loses (only for tiny a). The value is the exact chance of reaching a goal with the block in place.
 * Layout: as endgame_build.c (positions sorted by the sum of the free tiles in units of 2, then by rank within
 * the sum, the first free cell most significant). A move keeps the sum and a new tile adds 1 or 2 units, so sum s
 * reads only sums s+1 and s+2: built from the largest sum down, in place through a shared mapping.
 * Formats (one per table):
 *   u16  2 bytes a position: the value, v * 65535 rounded. The engine finds the best move by reading the
 *        afterstates (blk_after).
 *   mv8  1 byte a position: (q << 2) | d, d the position's best move (0 up, 1 right, 2 down, 3 left, the first
 *        of equals; 0 when no move has a chance) and q its value on 64 levels of the square root:
 *        q = round(63 sqrt(u / 65535)) of the u16 value u, kept in 1..62 unless u is 65535 or the chance is
 *        exactly 0 (a chance that rounds to u = 0 is q = 1, so q = 0 is the u16 engine's "no move"); read back
 *        as (q / 63)^2. The move is the u16 table's own argmax (the DP runs at u16 precision, below), so the
 *        engine plays exactly the u16 table's moves (up to the order among equal chances) with one read a
 *        reading; the value only meets the block_min thresholds. A value byte alone loses too much: along
 *        optimal play on b6 a9, quantized afterstate values change the u16 argmax at 23-35% of the decisions
 *        and lose 1e-4 to 5e-4 chance a decision (1.7-8% of the goal chance over a game), whatever the 8-bit
 *        scale (linear, sqrt, cbrt, log, symmetric log); the best scale on one table is poor on another.
 *        Built with u16 values for sums s+1 and s+2 kept in memory (three layers, 0.45 GB at b5 a10) and the
 *        mv8 byte written for sum s: the same DP as the u16 build, no u16 file. q of an mv8 file is exactly
 *        the 6-bit code of the u16 file's value at every position.
 *
 * Program:  block_build <b6|b5> <a1,a2,...> <outdir> [threads] [bytes 2|1]
 *           writes <outdir>/<shape>_a<a>.u16 (bytes 2, the default) or .mv8 (bytes 1) and <outdir>/block.txt
 *           ("<shape> <anchors> <free cells> <drops>[ <format per anchor>]", the formats written only when one is
 *           not u16; anchors merged with an earlier block.txt of the same shape; written last). Building an
 *           anchor deletes its file of the other format.
 *           block_build sizes  prints the size of every table.
 * Library:  tbb_load, tbb_load_fmt, tbb_value, tbb_move, tbb_size (for checks).
 */
#include <fcntl.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const char *name;
    int nblock, nfree;
    int block[6], free[11], drop[11];       /* free cell j holds at most 2^(a - drop[j]) */
} shape_t;

static const shape_t SHAPES[2] = {
    {"b6", 6, 10, {0, 1, 2, 4, 5, 6}, {3, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {1, 2, 1, 2, 3, 3, 3, 3, 3, 3}},
    {"b5", 5, 11, {0, 1, 2, 4, 5}, {3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {2, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3}},
};
#define AMIN 4                                   /* every cap at least 1 */
#define AMAX 15
enum { FMT_U16, FMT_MV8 };
static const char *FMT_NAME[2] = {"u16", "mv8"};
static const int FMT_BYTES[2] = {2, 1};

typedef struct {
    int sh, a, n, cap[11], maxs;
    uint64_t *offs, *pr[11], size, layer;    /* layer: the most positions of one sum */
    int fmt;
    uint16_t *v;                            /* u16: the table (built in place) */
    uint8_t *b8;                            /* mv8: the table */
    uint16_t *ring[3];                      /* mv8 build: u16 values of sums s, s+1, s+2 (layer s % 3) */
} tab_t;

static tab_t TAB[2][AMAX + 1];

static inline int units(int e) { return e ? 1 << (e - 1) : 0; }

/* the anchor of the block on board b (frame of the shape), or 0 if the block is not regular */
static int anchor(int sh, const uint8_t *b) {
    if (!b[0] || !b[1] || b[0] == b[1] || b[1] == b[2] || b[0] == b[4] || b[1] == b[5]) return 0;
    if (sh == 0) {
        if (b[4] == b[5] || b[5] == b[6] || b[2] <= b[6]) return 0;
        int a = b[4] < b[5] ? b[4] : b[5];
        return a < b[6] ? a : b[6];
    }
    if (b[5] == b[2] || b[4] <= b[5]) return 0;
    return b[5] < b[2] ? b[5] : b[2];
}

static int goal(int sh, const uint8_t *b, int a) {
    if (sh == 0) {
        int x = a - 1, y = a - 2, z = a - 3;
        return (b[8] == x && b[9] == y && ((b[10] == z && (b[14] == z || b[11] == z)) ||
                                           (b[13] == z && (b[12] == z || b[14] == z)))) ||
               (b[3] == x && b[7] == y && b[11] == z && (b[10] == z || b[15] == z));
    }
    int t = a - 1;
    return b[6] == t && ((b[7] == t - 1 && (b[3] == t - 1 || b[11] == t - 1)) ||
                         (b[3] == t - 1 && b[7] == t - 2 && b[11] == t - 2) ||
                         (b[10] == t - 1 && (b[9] == t - 1 || b[14] == t - 1 || b[11] == t - 1)));
}

static void caps_of(int sh, int a, int *cap) {
    for (int j = 0; j < SHAPES[sh].nfree; j++) cap[j] = a - SHAPES[sh].drop[j];
}

static uint64_t entries(int sh, int a) {
    int cap[11];
    uint64_t n = 1;
    caps_of(sh, a, cap);
    for (int j = 0; j < SHAPES[sh].nfree; j++) n *= (uint64_t)(cap[j] + 1);
    return n;
}

static int setup(tab_t *t, int sh, int a) {
    t->sh = sh; t->a = a; t->n = SHAPES[sh].nfree;
    caps_of(sh, a, t->cap);
    int n = t->n, S = 0;
    for (int j = 0; j < n; j++) S += units(t->cap[j]);
    t->maxs = S;
    uint64_t *cnt = calloc((size_t)(n + 1) * (S + 1), sizeof(uint64_t));   /* cells j .. n-1 with r units */
    t->offs = malloc((size_t)(S + 2) * sizeof(uint64_t));
    if (!cnt || !t->offs) { free(cnt); return -1; }
    cnt[(size_t)n * (S + 1)] = 1;
    for (int j = n - 1; j >= 0; j--)
        for (int r = 0; r <= S; r++) {
            uint64_t s = 0;
            for (int e = 0; e <= t->cap[j] && units(e) <= r; e++) s += cnt[(size_t)(j + 1) * (S + 1) + r - units(e)];
            cnt[(size_t)j * (S + 1) + r] = s;
        }
    t->offs[0] = 0;
    for (int s = 0; s <= S; s++) t->offs[s + 1] = t->offs[s] + cnt[s];
    t->size = t->offs[S + 1];
    t->layer = 0;
    for (int s = 0; s <= S; s++) if (cnt[s] > t->layer) t->layer = cnt[s];
    for (int j = 0; j < n; j++) {
        int C = t->cap[j] + 1;
        if (!(t->pr[j] = malloc((size_t)(S + 1) * C * sizeof(uint64_t)))) { free(cnt); return -1; }
        for (int r = 0; r <= S; r++) {
            uint64_t acc = 0;
            for (int v = 0; v < C; v++) {
                t->pr[j][(size_t)r * C + v] = acc;
                if (units(v) <= r) acc += cnt[(size_t)(j + 1) * (S + 1) + r - units(v)];
            }
        }
    }
    free(cnt);
    return 0;
}

static inline int fits(const tab_t *t, const uint8_t *b) {
    const int *F = SHAPES[t->sh].free;
    for (int j = 0; j < t->n; j++) if (b[F[j]] > t->cap[j]) return 0;
    return 1;
}

static inline uint64_t index_sum(const tab_t *t, const uint8_t *b, int *sum) {
    const int *F = SHAPES[t->sh].free;
    int s = 0;
    for (int j = 0; j < t->n; j++) s += units(b[F[j]]);
    *sum = s;
    uint64_t idx = t->offs[s];
    for (int j = 0; j < t->n; j++) {
        int v = b[F[j]];
        idx += t->pr[j][(size_t)s * (t->cap[j] + 1) + v];
        s -= units(v);
    }
    return idx;
}

static inline uint64_t index_of(const tab_t *t, const uint8_t *b) { int s; return index_sum(t, b, &s); }

/* the u16 value of position b: from the table, or while an mv8 table is built, from its layer in memory */
static inline uint16_t get16(const tab_t *t, const uint8_t *b) {
    if (t->ring[0]) {
        int s;
        uint64_t i = index_sum(t, b, &s);
        return t->ring[s % 3][i - t->offs[s]];
    }
    return t->v[index_of(t, b)];
}

static inline float get(const tab_t *t, const uint8_t *b) { return (float)get16(t, b) * (1.0f / 65535.0f); }

/* mv8: the 6-bit code of a u16 value (0 and 1 exact, any other value kept off them) and its value */
static inline int code6(uint16_t u) {
    if (u == 0 || u == 65535) return u ? 63 : 0;
    int q = (int)floor(63.0 * sqrt(u / 65535.0) + 0.5);
    return q < 1 ? 1 : q > 62 ? 62 : q;
}
static inline float value6(int q) { return (float)(q * q) * (1.0f / 3969.0f); }

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

/* dir: 0 up, 1 right, 2 down, 3 left (the engine's order) */
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

/* the value of position b (block cells hold walls that never merge) and its best move (0 if none has a chance) */
static float solve(const tab_t *t, const uint8_t *b, int *dir) {
    const shape_t *S = &SHAPES[t->sh];
    float best = 0.0f;
    *dir = 0;
    for (int d = 0; d < 4; d++) {
        uint8_t m[16];
        if (!move(b, d, m)) continue;
        int ok = 1;
        for (int i = 0; i < S->nblock && ok; i++) if (m[S->block[i]] != b[S->block[i]]) ok = 0;
        if (!ok || !fits(t, m)) continue;
        if (goal(t->sh, m, t->a)) { best = 1.0f; *dir = d; break; }
        double sum = 0.0; int empty = 0;
        for (int j = 0; j < t->n; j++) {
            int c = S->free[j];
            if (m[c]) continue;
            empty++;
            m[c] = 1; if (t->cap[j] >= 1) sum += 0.9 * get(t, m);
            m[c] = 2; if (t->cap[j] >= 2) sum += 0.1 * get(t, m);
            m[c] = 0;
        }
        float v = empty ? (float)(sum / empty) : 0.0f;
        if (v > best) { best = v; *dir = d; }
    }
    return best;
}

static void fill_rest(const tab_t *t, uint8_t *b, int j, int rem, const int *maxrem) {
    if (j == t->n) {
        if (rem) return;
        int s, d;
        uint64_t i = index_sum(t, b, &s);
        float v = solve(t, b, &d);
        uint16_t u = (uint16_t)(v * 65535.0f + 0.5f);
        if (t->fmt == FMT_U16) t->v[i] = u;
        else {                                /* a chance below u16's half step is still a chance: code 1 */
            t->ring[s % 3][i - t->offs[s]] = u;
            t->b8[i] = (uint8_t)((u || v <= 0.0f ? code6(u) : 1) << 2 | d);
        }
        return;
    }
    if (rem > maxrem[j]) return;
    int c = SHAPES[t->sh].free[j];
    for (int e = 0; e <= t->cap[j] && units(e) <= rem; e++) {
        b[c] = (uint8_t)e;
        fill_rest(t, b, j + 1, rem - units(e), maxrem);
    }
    b[c] = 0;
}

static void walls(int sh, uint8_t *b) {               /* distinct, above every free tile: never merge */
    memset(b, 0, 16);
    for (int i = 0; i < SHAPES[sh].nblock; i++) b[SHAPES[sh].block[i]] = (uint8_t)(16 + i);
}

static void build_table(tab_t *t, int threads) {
    int n = t->n, maxrem[12];
    maxrem[n] = 0;
    for (int j = n - 1; j >= 0; j--) maxrem[j] = maxrem[j + 1] + units(t->cap[j]);
    int c0 = t->cap[0] + 1, c1 = t->cap[1] + 1, tasks = c0 * c1;
    const int *F = SHAPES[t->sh].free;
    time_t t0 = time(NULL), last = t0;
    (void)threads;
    for (int s = maxrem[0]; s >= 0; s--) {
        #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (int k = 0; k < tasks; k++) {
            int e0 = k % c0, e1 = k / c0, rem = s - units(e0) - units(e1);
            if (rem < 0 || rem > maxrem[2]) continue;
            uint8_t b[16];
            walls(t->sh, b);
            b[F[0]] = (uint8_t)e0;
            b[F[1]] = (uint8_t)e1;
            fill_rest(t, b, 2, rem, maxrem);
        }
        if (time(NULL) - last >= 60) {
            last = time(NULL);
            fprintf(stderr, "  %s a %d: sum %d done (%.1f%% of positions), %lds\n", SHAPES[t->sh].name, t->a, 2 * s,
                    100.0 * (double)(t->size - t->offs[s]) / (double)t->size, (long)(last - t0));
        }
    }
}

static void *map_file(const char *path, uint64_t bytes, int write) {
    int fd = open(path, write ? O_RDWR | O_CREAT | O_TRUNC : O_RDONLY, 0644);
    if (fd < 0) return NULL;
    if (write && ftruncate(fd, (off_t)bytes) != 0) { close(fd); return NULL; }
    if (!write && lseek(fd, 0, SEEK_END) != (off_t)bytes) { close(fd); return NULL; }
    void *m = mmap(NULL, bytes, write ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return m == MAP_FAILED ? NULL : m;
}

static void table_path(char *path, size_t n, const char *dir, int sh, int a, int fmt) {
    snprintf(path, n, "%s/%s_a%d.%s", dir, SHAPES[sh].name, a, FMT_NAME[fmt]);
}

static int shape_of(const char *name) {
    for (int sh = 0; sh < 2; sh++) if (strcmp(name, SHAPES[sh].name) == 0) return sh;
    return -1;
}

/* the format of each anchor listed in <dir>/block.txt for shape sh (-1: not listed) */
static void read_listed(const char *dir, int sh, int *fmt) {
    char path[4096], line[1024], name[16], list[256], cells[256], drops[256], fmts[512];
    for (int a = 0; a <= AMAX; a++) fmt[a] = -1;
    snprintf(path, sizeof path, "%s/block.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) return;
    int n = fgets(line, sizeof line, f) ? sscanf(line, "%15s %255s %255s %255s %511s", name, list, cells, drops, fmts) : 0;
    fclose(f);
    if (n < 2 || strcmp(name, SHAPES[sh].name) != 0) return;
    char *q = n == 5 ? fmts : NULL;
    for (char *p = list; *p;) {
        int a = atoi(p), k = FMT_U16;
        if (q && *q) {
            if (strncmp(q, FMT_NAME[FMT_MV8], 3) == 0) k = FMT_MV8;
            while (*q && *q != ',') q++;
            if (*q) q++;
        }
        if (a >= AMIN && a <= AMAX) fmt[a] = k;
        while (*p && *p != ',') p++;
        if (*p) p++;
    }
}

/* The tables of shape sh for the anchors listed in format fmt (and those of an earlier block.txt of the same
 * shape, in their formats). */
int tbb_build_fmt(int sh, const int *anchors, int na, const char *dir, int threads, int fmt) {
    char path[4096];
    int have[AMAX + 1];
    if (fmt != FMT_U16 && fmt != FMT_MV8) return -1;
    read_listed(dir, sh, have);                       /* earlier tables of this shape stay listed */
    snprintf(path, sizeof path, "%s/block.txt", dir);
    unlink(path);                                     /* written last: a directory without it is not a finished build */
    for (int i = 0; i < na; i++) {
        int a = anchors[i];
        tab_t *t = &TAB[sh][a];
        if (setup(t, sh, a)) return -1;
        table_path(path, sizeof path, dir, sh, a, 1 - fmt);
        unlink(path);                                 /* the same table in the other format */
        table_path(path, sizeof path, dir, sh, a, fmt);
        uint64_t bytes = t->size * FMT_BYTES[fmt];
        void *m = map_file(path, bytes, 1);
        if (!m) return -1;
        t->fmt = fmt;
        if (fmt == FMT_U16) t->v = m;
        else {
            t->b8 = m;
            for (int r = 0; r < 3; r++)
                if (!(t->ring[r] = malloc(t->layer * sizeof(uint16_t)))) return -1;
        }
        time_t t0 = time(NULL);
        build_table(t, threads);
        msync(m, bytes, MS_SYNC);
        fprintf(stderr, "%s a %d: %llu positions, %lds\n", SHAPES[sh].name, a, (unsigned long long)t->size,
                (long)(time(NULL) - t0));
        munmap(m, bytes);
        for (int r = 0; r < 3; r++) { free(t->ring[r]); t->ring[r] = NULL; }
        t->v = NULL;
        t->b8 = NULL;
        have[a] = fmt;
    }
    int mixed = 0;
    for (int a = AMIN; a <= AMAX; a++) if (have[a] > FMT_U16) mixed = 1;
    snprintf(path, sizeof path, "%s/block.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%s ", SHAPES[sh].name);
    for (int a = AMIN, first = 1; a <= AMAX; a++) if (have[a] >= 0) { fprintf(f, first ? "%d" : ",%d", a); first = 0; }
    for (int j = 0; j < SHAPES[sh].nfree; j++) fprintf(f, j ? ",%d" : " %d", SHAPES[sh].free[j]);
    for (int j = 0; j < SHAPES[sh].nfree; j++) fprintf(f, j ? ",%d" : " %d", SHAPES[sh].drop[j]);
    if (mixed)
        for (int a = AMIN, first = 1; a <= AMAX; a++)
            if (have[a] >= 0) { fprintf(f, first ? " %s" : ",%s", FMT_NAME[have[a]]); first = 0; }
    fprintf(f, "\n");
    return fclose(f) == 0 ? 0 : -1;
}

int tbb_build(int sh, const int *anchors, int na, const char *dir, int threads) {
    return tbb_build_fmt(sh, anchors, na, dir, threads, FMT_U16);
}

/* table (sh, a) in format fmt (0 u16, 1 mv8) from dir, read-only */
int tbb_load_fmt(const char *dir, int sh, int a, int fmt) {
    if (sh < 0 || sh > 1 || a < AMIN || a > AMAX || fmt < 0 || fmt > 1) return -1;
    tab_t *t = &TAB[sh][a];
    char path[4096];
    if (setup(t, sh, a)) return -1;
    table_path(path, sizeof path, dir, sh, a, fmt);
    void *m = map_file(path, t->size * FMT_BYTES[fmt], 0);
    if (!m) return -1;
    t->fmt = fmt;
    if (fmt == FMT_U16) t->v = m; else t->b8 = m;
    return 0;
}

int tbb_load(const char *dir, int sh, int a) { return tbb_load_fmt(dir, sh, a, FMT_U16); }

unsigned long long tbb_size(int sh, int a) { return (unsigned long long)entries(sh, a); }

static const tab_t *loaded_for(const uint8_t *b, int sh, int a) {
    if (sh < 0 || sh > 1 || a < AMIN || a > AMAX || !(TAB[sh][a].v || TAB[sh][a].b8)) return NULL;
    if (anchor(sh, b) < a || !fits(&TAB[sh][a], b)) return NULL;
    return &TAB[sh][a];
}

/* board: 16 exponents, the block in the shape's place; -1 if it is not a position of table (sh, a): the block
 * not regular, its anchor below a, or a free tile above its cap. mv8: the 6-bit value. */
float tbb_value(const uint8_t *b, int sh, int a) {
    const tab_t *t = loaded_for(b, sh, a);
    if (!t) return -1.0f;
    return t->fmt == FMT_MV8 ? value6(t->b8[index_of(t, b)] >> 2) : get(t, b);
}

/* mv8: the stored best move of board b (table frame), -2 if the value is 0; -1 as tbb_value or a u16 table */
int tbb_move(const uint8_t *b, int sh, int a) {
    const tab_t *t = loaded_for(b, sh, a);
    if (!t || t->fmt != FMT_MV8) return -1;
    uint8_t e = t->b8[index_of(t, b)];
    return e >> 2 ? e & 3 : -2;
}

#ifdef EGB_MAIN
static void print_sizes(int sh, const int *anchors, int na, int bytes) {
    for (int i = 0; i < na; i++) {
        uint64_t n = entries(sh, anchors[i]), lay;
        tab_t t = {0};
        setup(&t, sh, anchors[i]);                   /* small: the layer sizes */
        lay = t.layer;
        free(t.offs);
        for (int j = 0; j < t.n; j++) free(t.pr[j]);
        fprintf(stderr, "  %s a %2d (anchor %6d): %14llu positions, %8.2f GB", SHAPES[sh].name, anchors[i],
                1 << anchors[i], (unsigned long long)n, (double)bytes * (double)n / 1e9);
        if (bytes == 1) fprintf(stderr, " (build memory %.2f GB: 3 layers of u16)", 6.0 * (double)lay / 1e9);
        fprintf(stderr, "\n");
    }
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "sizes") == 0) {
        int all[] = {4, 5, 6, 7, 8, 9, 10, 11, 12};
        for (int sh = 0; sh < 2; sh++) print_sizes(sh, all, 9, 2);
        return 0;
    }
    if (argc < 4) {
        fprintf(stderr, "block_build <b6|b5> <a1,a2,...> <outdir> [threads] [bytes 2|1]   |   block_build sizes\n");
        return 2;
    }
    int sh = shape_of(argv[1]), anchors[16], na = 0, threads = argc > 4 ? atoi(argv[4]) : 1;
    int bytes = argc > 5 ? atoi(argv[5]) : 2, fmt = bytes == 1 ? FMT_MV8 : FMT_U16;
    if (sh < 0) { fprintf(stderr, "shape must be b6 or b5\n"); return 2; }
    if (bytes != 1 && bytes != 2) { fprintf(stderr, "bytes must be 2 (u16) or 1 (mv8)\n"); return 2; }
    for (char *p = argv[2]; *p && na < 16;) {
        int a = atoi(p);
        if (a < AMIN || a > AMAX) { fprintf(stderr, "anchors must be %d .. %d\n", AMIN, AMAX); return 2; }
        anchors[na++] = a;
        while (*p && *p != ',') p++;
        if (*p) p++;
    }
    uint64_t total = 0;
    for (int i = 0; i < na; i++) total += (uint64_t)bytes * entries(sh, anchors[i]);
    fprintf(stderr, "to build (%s):\n", FMT_NAME[fmt]);
    print_sizes(sh, anchors, na, bytes);
    fprintf(stderr, "  total %.2f GB\n", (double)total / 1e9);
    struct statvfs fs;
    if (statvfs(argv[3], &fs) == 0) {
        uint64_t avail = (uint64_t)fs.f_bavail * fs.f_frsize, old = 0;
        char path[4096];
        for (int i = 0; i < na; i++)                     /* files rebuilt in place (either format) free their space first */
            for (int k = 0; k < 2; k++) {
                table_path(path, sizeof path, argv[3], sh, anchors[i], k);
                FILE *f = fopen(path, "rb");
                if (f) { fseek(f, 0, SEEK_END); old += (uint64_t)ftell(f); fclose(f); }
            }
        if (total > avail + old) {
            fprintf(stderr, "not enough disk: %.2f GB needed, %.2f GB free\n", (double)total / 1e9,
                    (double)(avail + old) / 1e9);
            return 1;
        }
    }
    if (tbb_build_fmt(sh, anchors, na, argv[3], threads, fmt)) {
        fprintf(stderr, "build failed (memory or files)\n");
        return 1;
    }
    return 0;
}
#endif
