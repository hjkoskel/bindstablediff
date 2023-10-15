
#ifdef __cplusplus
#include <vector>
#include <string>
#include "stable-diffusion.h"

extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

int printsysteminfo();

typedef struct{
    char *modelfilename;
    int n_threads;
    void *sd; //Actual pointer to class
}StableDiffusionModel;

int loadStableDiffusion(char *sdfilename,int n_threads,int enumSchedule, StableDiffusionModel *model);
int freeStableDiffusionModel(StableDiffusionModel *model);

uint8_t *txt2img(StableDiffusionModel *model,
    char *prompt,
    char *negativePrompt,
    float cfg_scale,
    int width,int height, 
    int sampleMethod,
    int sampleSteps,
    int64_t seed);

uint8_t *img2img(StableDiffusionModel *model,
    uint8_t *initialImage,
    char *prompt,
    char *negativePrompt,
    float cfg_scale,
    int width,int height, 
    int sampleMethod,
    int sampleSteps,
    float strength,
    int64_t seed);


#ifdef __cplusplus
}

//std::vector<std::string> create_vector(const char** strings, int count);
//void delete_vector(std::vector<std::string>* vec);
#endif
