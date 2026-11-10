/* host_post.c — desktop build of the app's SHARP C core, for tools/compare_outputs.py.
 *
 * Runs exactly the same post-processing the app runs on-device (sharp_post:
 * depth_from_disparity -> initializer -> composer -> unproject -> save_ply), so a
 * desktop reference .ply differs from the device .ply only by model outputs.
 *
 *   host_post prep  <image.jpg> <out image.raw>        prints: fpx origW origH
 *   host_post pre   <image.raw> <workDir>               35 patch_p*.raw + x2.raw
 *   host_post merge <workDir> <peOutDir> <ieOutDir>     6 merged feature raws
 *   host_post post  <workDir> <fpx> <origW> <origH> <out.ply>
 *       workDir must contain image.raw, disparity.raw, delta.raw (float32 NCHW)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sharp_pipeline.h"

int main(int argc, char **argv)
{
    if (argc == 4 && strcmp(argv[1], "prep") == 0) {
        float fpx = 0, df = 0; int w = 0, h = 0;
        int rc = sharp_prep_image(argv[2], argv[3], &fpx, &df, &w, &h);
        if (rc == 0) printf("%.6f %d %d\n", fpx, w, h);
        return rc;
    }
    if (argc == 4 && strcmp(argv[1], "pre") == 0) return sharp_pre(argv[2], argv[3]);
    if (argc == 5 && strcmp(argv[1], "merge") == 0) return sharp_merge(argv[2], argv[3], argv[4]);
    if (argc == 7 && strcmp(argv[1], "post") == 0) {
        int rc = sharp_post(argv[2], (float)atof(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6]);
        fprintf(stderr, "sharp_post rc=%d\n", rc);
        return rc;
    }
    fprintf(stderr, "usage: host_post prep|pre|merge|post ... (see source header)\n");
    return 2;
}
