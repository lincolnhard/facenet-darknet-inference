#include "darknet.h"

static network *net;

void init_net
    (
    const char *cfgfile,
    const char *weightfile,
    int *inw,
    int *inh,
    int *outc
    )
{
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    *inw = net->w;
    *inh = net->h;
    *outc = net->layers[net->n - 1].out_c;

    nnp_initialize();
    net->threadpool = pthreadpool_create(4);
}

float *run_net
    (
    float *indata
    )
{
    network_predict(net, indata);
    return net->output;
}

void free_net
    (
    void
    )
{
    free_network(net);

    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
}
