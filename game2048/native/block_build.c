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
 * Values: uint16, v * 65535 rounded.
 *
 * Program:  block_build <b6|b5> <a1,a2,...> <outdir> [threads]
 *           writes <outdir>/<shape>_a<a>.u16 and <outdir>/block.txt ("<shape> <anchors> <free cells> <drops>",
 *           anchors merged with an earlier block.txt of the same shape; written last).
 *           block_build sizes  prints the size of every table.
 * Library:  tbb_load, tbb_value, tbb_size (for checks).
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

typedef struct {
    int sh, a, n, cap[11], maxs;
    uint64_t *offs, *pr[11], size;
    uint16_t *v;
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

static inline uint64_t index_of(const tab_t *t, const uint8_t *b) {
    const int *F = SHAPES[t->sh].free;
    int s = 0;
    for (int j = 0; j < t->n; j++) s += units(b[F[j]]);
    uint64_t idx = t->offs[s];
    for (int j = 0; j < t->n; j++) {
        int v = b[F[j]];
        idx += t->pr[j][(size_t)s * (t->cap[j] + 1) + v];
        s -= units(v);
    }
    return idx;
}

static inline float get(const tab_t *t, const uint8_t *b) { return (float)t->v[index_of(t, b)] * (1.0f / 65535.0f); }

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

/* the value of position b (block cells hold walls that never merge) */
static float solve(const tab_t *t, const uint8_t *b) {
    const shape_t *S = &SHAPES[t->sh];
    float best = 0.0f;
    for (int d = 0; d < 4; d++) {
        uint8_t m[16];
        if (!move(b, d, m)) continue;
        int ok = 1;
        for (int i = 0; i < S->nblock && ok; i++) if (m[S->block[i]] != b[S->block[i]]) ok = 0;
        if (!ok || !fits(t, m)) continue;
        if (goal(t->sh, m, t->a)) { best = 1.0f; break; }
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
        if (v > best) best = v;
    }
    return best;
}

static void fill_rest(const tab_t *t, uint8_t *b, int j, int rem, const int *maxrem) {
    if (j == t->n) {
        if (rem) return;
        t->v[index_of(t, b)] = (uint16_t)(solve(t, b) * 65535.0f + 0.5f);
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

static uint16_t *map_file(const char *path, uint64_t n, int write) {
    int fd = open(path, write ? O_RDWR | O_CREAT | O_TRUNC : O_RDONLY, 0644);
    if (fd < 0) return NULL;
    if (write && ftruncate(fd, (off_t)(n * 2)) != 0) { close(fd); return NULL; }
    if (!write && lseek(fd, 0, SEEK_END) != (off_t)(n * 2)) { close(fd); return NULL; }
    void *m = mmap(NULL, n * 2, write ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return m == MAP_FAILED ? NULL : (uint16_t *)m;
}

static int shape_of(const char *name) {
    for (int sh = 0; sh < 2; sh++) if (strcmp(name, SHAPES[sh].name) == 0) return sh;
    return -1;
}

/* The tables of shape sh for the anchors listed (and those of an earlier block.txt of the same shape). */
int tbb_build(int sh, const int *anchors, int na, const char *dir, int threads) {
    char path[4096];
    int have[AMAX + 1] = {0};
    snprintf(path, sizeof path, "%s/block.txt", dir);
    FILE *f = fopen(path, "r");
    if (f) {                                          /* earlier tables of this shape stay listed */
        char name[16], list[256];
        if (fscanf(f, "%15s %255s", name, list) == 2 && strcmp(name, SHAPES[sh].name) == 0)
            for (char *p = list; *p;) {
                int a = atoi(p);
                if (a >= AMIN && a <= AMAX) have[a] = 1;
                while (*p && *p != ',') p++;
                if (*p) p++;
            }
        fclose(f);
    }
    unlink(path);                                     /* written last: a directory without it is not a finished build */
    for (int i = 0; i < na; i++) {
        int a = anchors[i];
        tab_t *t = &TAB[sh][a];
        if (setup(t, sh, a)) return -1;
        snprintf(path, sizeof path, "%s/%s_a%d.u16", dir, SHAPES[sh].name, a);
        if (!(t->v = map_file(path, t->size, 1))) return -1;
        time_t t0 = time(NULL);
        build_table(t, threads);
        msync(t->v, t->size * 2, MS_SYNC);
        fprintf(stderr, "%s a %d: %llu positions, %lds\n", SHAPES[sh].name, a, (unsigned long long)t->size,
                (long)(time(NULL) - t0));
        munmap(t->v, t->size * 2);
        t->v = NULL;
        have[a] = 1;
    }
    snprintf(path, sizeof path, "%s/block.txt", dir);
    if (!(f = fopen(path, "w"))) return -1;
    fprintf(f, "%s ", SHAPES[sh].name);
    for (int a = AMIN, first = 1; a <= AMAX; a++) if (have[a]) { fprintf(f, first ? "%d" : ",%d", a); first = 0; }
    for (int j = 0; j < SHAPES[sh].nfree; j++) fprintf(f, j ? ",%d" : " %d", SHAPES[sh].free[j]);
    for (int j = 0; j < SHAPES[sh].nfree; j++) fprintf(f, j ? ",%d" : " %d", SHAPES[sh].drop[j]);
    fprintf(f, "\n");
    return fclose(f) == 0 ? 0 : -1;
}

int tbb_load(const char *dir, int sh, int a) {
    if (sh < 0 || sh > 1 || a < AMIN || a > AMAX) return -1;
    tab_t *t = &TAB[sh][a];
    char path[4096];
    if (setup(t, sh, a)) return -1;
    snprintf(path, sizeof path, "%s/%s_a%d.u16", dir, SHAPES[sh].name, a);
    return (t->v = map_file(path, t->size, 0)) ? 0 : -1;
}

unsigned long long tbb_size(int sh, int a) { return (unsigned long long)entries(sh, a); }

/* board: 16 exponents, the block in the shape's place; -1 if it is not a position of table (sh, a): the block
 * not regular, its anchor below a, or a free tile above its cap */
float tbb_value(const uint8_t *b, int sh, int a) {
    if (sh < 0 || sh > 1 || a < AMIN || a > AMAX || !TAB[sh][a].v) return -1.0f;
    if (anchor(sh, b) < a || !fits(&TAB[sh][a], b)) return -1.0f;
    return get(&TAB[sh][a], b);
}

#ifdef EGB_MAIN
static void print_sizes(int sh, const int *anchors, int na) {
    for (int i = 0; i < na; i++) {
        uint64_t n = entries(sh, anchors[i]);
        fprintf(stderr, "  %s a %2d (anchor %6d): %14llu positions, %8.2f GB\n", SHAPES[sh].name, anchors[i],
                1 << anchors[i], (unsigned long long)n, 2.0 * (double)n / 1e9);
    }
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "sizes") == 0) {
        int all[] = {4, 5, 6, 7, 8, 9, 10, 11, 12};
        for (int sh = 0; sh < 2; sh++) print_sizes(sh, all, 9);
        return 0;
    }
    if (argc < 4) { fprintf(stderr, "block_build <b6|b5> <a1,a2,...> <outdir> [threads]   |   block_build sizes\n"); return 2; }
    int sh = shape_of(argv[1]), anchors[16], na = 0, threads = argc > 4 ? atoi(argv[4]) : 1;
    if (sh < 0) { fprintf(stderr, "shape must be b6 or b5\n"); return 2; }
    for (char *p = argv[2]; *p && na < 16;) {
        int a = atoi(p);
        if (a < AMIN || a > AMAX) { fprintf(stderr, "anchors must be %d .. %d\n", AMIN, AMAX); return 2; }
        anchors[na++] = a;
        while (*p && *p != ',') p++;
        if (*p) p++;
    }
    uint64_t total = 0;
    for (int i = 0; i < na; i++) total += 2 * entries(sh, anchors[i]);
    fprintf(stderr, "to build:\n");
    print_sizes(sh, anchors, na);
    fprintf(stderr, "  total %.2f GB\n", (double)total / 1e9);
    struct statvfs fs;
    if (statvfs(argv[3], &fs) == 0) {
        uint64_t avail = (uint64_t)fs.f_bavail * fs.f_frsize, old = 0;
        char path[4096];
        for (int i = 0; i < na; i++) {                   /* files rebuilt in place free their space first */
            snprintf(path, sizeof path, "%s/%s_a%d.u16", argv[3], SHAPES[sh].name, anchors[i]);
            FILE *f = fopen(path, "rb");
            if (f) { fseek(f, 0, SEEK_END); old += (uint64_t)ftell(f); fclose(f); }
        }
        if (total > avail + old) {
            fprintf(stderr, "not enough disk: %.2f GB needed, %.2f GB free\n", (double)total / 1e9,
                    (double)(avail + old) / 1e9);
            return 1;
        }
    }
    if (tbb_build(sh, anchors, na, argv[3], threads)) { fprintf(stderr, "build failed (memory or files)\n"); return 1; }
    return 0;
}
#endif
