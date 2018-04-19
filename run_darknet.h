#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

#ifdef __cplusplus
extern "C"{
#endif

void init_net
    (
    const char *cfgfile,
    const char *weightfile,
    int *inw,
    int *inh,
    int *outc
    );

float *run_net
    (
    float *indata
    );

void free_net
    (
    void
    );

#ifdef __cplusplus
}
#endif

#endif // RUN_DARKNET_H
