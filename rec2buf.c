#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define SYNC    (-16711936)  // 0xFF00FF00

typedef struct {
    int         sync;
    int         ver;
    int         bl_no;
    int         mjd;
    double      sec;
    int         cfg_idx;
    int         src_idx;
    int         freq_idx;
    char        polar[2];
    int         pulsarbin;
    double      weight;
    double      uvw[3];
}SwinHeader;

void print_header(SwinHeader *p){
    printf("sync:       %d\n", p->sync);
    printf("bl_no:      %d\n", p->bl_no);
    printf("polar:      %c%c\n", p->polar[0], p->polar[1]);
    printf("\n\n");
}

double dt(int mjd0, double sec0, int mjd, double sec){
    
    return (mjd - mjd0) * 86400. + sec - sec0;
}

int pol2idx(char *p){
    
    if(p[0] == 'L' && p[1] == 'L') return 0;
    if(p[0] == 'R' && p[1] == 'R') return 1;
    if(p[0] == 'L' && p[1] == 'R') return 2;
    if(p[0] == 'R' && p[1] == 'L') return 3;

    return -1;
}

void rec2buf(int bl_no, void *_rec, long nrec, 
            int mjd0, double sec0, double t0, double tap, 
            int nap, int nfreq, int nvis, 
            void *_fidx,
            void *_b_LL, void *_b_RR, void *_b_LR, void *_b_RL){

//    printf("bl_no:      %d\n", bl_no);
//    printf("nrec:       %ld\n", nrec);
//    printf("nap: %d, nfreq: %d, nvis: %d\n", nap, nfreq, nvis);
//    printf("\n\n");

    char *p   =   (char *)_rec;
    char *buf[4];
    buf[0]  =   (char *)_b_LL;
    buf[1]  =   (char *)_b_RR;
    buf[2]  =   (char *)_b_LR;
    buf[3]  =   (char *)_b_RL;

    int *fidx    =   (int *)_fidx;
    
// header size is not sizeof(SwinHeader)!
    int size_hdr    =   74;
    int size_data   =   nvis * 4 * 2;
    int size_rec    =   size_hdr + size_data;

//    printf("size of hdr from C: %d\n\n", size_hdr);
//    printf("size of rec from C: %d\n\n", size_rec);
 
    long i;
    int pol_idx, f_idx, ap_idx, idx;
    double t_rec;
    SwinHeader *ph;
//    nrec    =   10;
//#pragma omp parallel for num_threads(4)
    for(i = 0; i < nrec; ++i){

        ph  =   (SwinHeader *)(p + i * size_rec);

//        print_header(ph);

        if(ph->bl_no != bl_no) continue;

        assert(ph->sync == SYNC);

        pol_idx =   pol2idx(ph->polar);
        assert(pol_idx >=0);
        if(buf[pol_idx] == NULL) continue;

        f_idx   =   fidx[ph->freq_idx];
        if(f_idx < 0) continue;
        
        t_rec   =   dt(mjd0, sec0, ph->mjd, ph->sec);
        if(t_rec < t0) continue;
        ap_idx  =   (int)((t_rec - t0) / tap);
        if(ap_idx >= nap) continue;
        
        idx =   (ap_idx * nfreq + f_idx);
        memcpy( (void *)(buf[pol_idx] + idx * size_data), 
                (void *)(p + i * size_rec + size_hdr), size_data);
    } // for i < nrec
} 
