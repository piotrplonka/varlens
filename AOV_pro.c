#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NMAX 25000

/* C implementation of analysis of variance.
 * 
 * P. Mroz @ Caltech, 6 Mar 2020
 */

#define median(a,n) kth_smallest(a,n,(((n)&1)?((n)/2):(((n)/2)-1)))

double AOV (double freq, double *t, double *m, double avg, int npts, int r) 
{    
    int i,idx,n[r];
    double aux,s1,s2,F,phase[NMAX],sum1[r],sum2[r];
    
    /* calculate orbital phase */
    for (i=0; i<npts; i++)
    {
        aux = (t[i]-t[0])*freq;
        phase[i] = aux-(int)aux;
    }
    // calculate mean and variance in each phase bin [ 0 .. r-1 ]
    for (i=0; i<r; i++) 
    {
        n[i] = 0;
        sum1[i] = 0.0;
        sum2[i] = 0.0;
    }
    for (i=0; i<npts; i++) 
    {
        idx = (int)(phase[i]*r);
        sum1[idx] += m[i];
        sum2[idx] += m[i]*m[i];
        n[idx] += 1;
    }
    s1 = 0.0; s2 = 0.0;
    for (i=0; i<r; i++) {
        if (n[i] == 0) continue;
        sum1[i] /= (double)n[i]; // mean in each phase bin
        s1 += n[i]*(sum1[i]-avg)*(sum1[i]-avg);
        s2 += sum2[i]-n[i]*sum1[i]*sum1[i];
    }
    F = s1/s2;
    F *= (double)(npts-r);
    F /= (double)(r-1);
    
    return F;
}

int remove_lowest_datapoint (double *time, double *mag, double *magerr, int npts) {
    int i, idx;
    double tmp;
    tmp = 0.0;
    idx = 0;
    for (i=0; i< npts; i++) {
      if (mag[i] > tmp) {
        tmp = mag[i];
        idx = i;
      }
    }
    for (i=idx; i<(npts-1); i++) {
      time[i] = time[i+1];
      mag[i] = mag[i+1];
      magerr[i] = magerr[i+1];
    }
    return (npts-1);
}

int read_data (char *filename, double *time, double *flux, double *ferr) {
    
    FILE *fp;
    int i;
    
    fp = fopen(filename,"r");
    if (fp == NULL) {
        fprintf(stderr,"Error while opening file %s\n",filename);
        return 0;
    }
    i = 0;
    while (fscanf(fp,"%lf %lf %lf",&time[i],&flux[i],&ferr[i]) != EOF) i++;
    fclose(fp);
    
    return i;
    
}

double kth_smallest(double *a, int n, int k) {
    // See: http://ndevilla.free.fr/median/median/
    int i,j,l,m ;
    double x, tmp ;
    double *copy;
    
    copy = (double *) malloc(sizeof(double)*n);
    for (i=0; i<n; i++) copy[i] = a[i];

    l=0 ; m=n-1 ;
    while (l<m) {
        x=copy[k] ;
        i=l ;
        j=m ;
        do {
            while (copy[i]<x) i++ ;
            while (x<copy[j]) j-- ;
            if (i<=j) {
                tmp = copy[i];
                copy[i] = copy[j];
                copy[j] = tmp;
                i++ ; j-- ;
            }
        } while (i<=j) ;
        if (j<k) l=i ;
        if (k<i) m=j ;
    }
    tmp = copy[k];
    
    free(copy);
    return tmp;
}

int main (int argc, char *argv[]) 
{

    double time[NMAX],flux[NMAX],ferr[NMAX],freq_min,freq_max;
    double delta_freq,freq,power,power_max,freq_best,mean_flux;
    int npts,n_freq,i;
    
    if (argc != 2) {
        fprintf(stderr,"Usage: AOV filename\n");
        return 1;
    }
    
    npts = read_data(argv[1],time,flux,ferr);
    if (npts < 30) return(0);
    npts = remove_lowest_datapoint(time, flux, ferr, npts);
    
    freq_min = 0.01;
    freq_max = 250.0;
    delta_freq = 0.002*freq_min;
    n_freq = (int)((freq_max-freq_min)/delta_freq);
    
    mean_flux = 0.0;
    for (i=0; i<npts; i++) mean_flux += flux[i];
    mean_flux /= (double)npts;
    

    power_max = 0.0;
    freq = freq_min;
    for (i=0; i<n_freq; i++) {
        power = AOV(freq,time,flux,mean_flux,npts,10);
        if (power > power_max) {
            power_max = power;
            freq_best = freq;
        }
        freq += delta_freq;
    }
    
    printf("%-20s %16.12f %8.3f %4d\n",argv[1],1.0/freq_best,power_max,npts);

    return 0;
    
    
}
