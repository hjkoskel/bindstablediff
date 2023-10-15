#include "bindstablediff.h"

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
            _exit(130);
    }
}
#endif


#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include <cstring>

int printsysteminfo(){
    printf("%s", sd_get_system_info().c_str());
    return 0;
}

int loadStableDiffusion(char *sdfilename,int n_threads,int enumSchedule, StableDiffusionModel *model){
    printf("going to init stable diffusion from file %s (enumSchedule=%d)\n",sdfilename,enumSchedule);
    bool vae_decode_only = false; // IMG2IMG tää on false :( meneepä mutkikkaaksi!  EI kun true aina!
    bool free_params_immediately = false;
    model->modelfilename=sdfilename;
    model->sd = new StableDiffusion(n_threads, vae_decode_only, free_params_immediately,STD_DEFAULT_RNG);
    
    std::string sFname(sdfilename);
    StableDiffusion * s= static_cast<StableDiffusion *>(model->sd);
    s->load_from_file(sFname, (Schedule)enumSchedule);
    return 0;
}

//Simple and dummy way to use model with no real control to output
uint8_t *txt2img(StableDiffusionModel *model,
    char *prompt,
    char *negativePrompt,
    float cfg_scale,
    int width,int height, 
    int sampleMethod,
    int sampleSteps,
    int64_t seed){

   std::string sPrompt(prompt);
   std::string sNegativePrompt(negativePrompt);
 
   StableDiffusion * theModel= static_cast<StableDiffusion *>(model->sd);

   std::vector<uint8_t> resultVec= theModel->txt2img(
        sPrompt,
        sNegativePrompt,
        cfg_scale,
        width, height,
        (SampleMethod)sampleMethod,
        sampleSteps,
        seed);

    uint8_t *resultData=(uint8_t *)calloc(resultVec.size(),1);
    std::memcpy(resultData,&resultVec[0],resultVec.size());
    //printf("copied vector size of %d\n",resultVec.size());
    return resultData;
}

//TBD  img2img have its issues
uint8_t *img2img(StableDiffusionModel *model,
    uint8_t *initialImage,
    char *prompt,
    char *negativePrompt,
    float cfg_scale,
    int width,int height, 
    int sampleMethod,
    int sampleSteps,
    float strength,
    int64_t seed){

    printf("\n\nPROMPT %s\n",prompt);
    printf("NEGATIVE PROMPT %s\n",negativePrompt);
    printf("cfg_scale=%f\n",cfg_scale);
    printf("sample_method=%d\n",sampleMethod);
    printf("sample_steps=%d\n",sampleSteps);
    printf("strength=%f\n",strength);
    printf("seed=%ld\n\n",seed);

    std::vector<uint8_t> initImgVec(initialImage, initialImage + (width*height*3));

   std::string sPrompt(prompt);
   std::string sNegativePrompt(negativePrompt);
 
   StableDiffusion * theModel= static_cast<StableDiffusion *>(model->sd);

   std::vector<uint8_t> resultVec= theModel->img2img(
    initImgVec,
        sPrompt,
        sNegativePrompt,
        cfg_scale,
        width, height,
        (SampleMethod)sampleMethod,
        sampleSteps,
        strength,
        seed);

    uint8_t *resultData=(uint8_t *)calloc(resultVec.size(),1);
    std::memcpy(resultData,&resultVec[0],resultVec.size());
    return resultData;
}


int freeStableDiffusionModel(StableDiffusionModel *model){
    //TODO IMPLEMENT
    StableDiffusion * s= static_cast<StableDiffusion *>(model->sd);
    delete(s);
    return 0;
}
