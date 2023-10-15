# stbdif

Stbdif is stable diffusion program meant for batch generating stable diffusion images on CPU.
It is not as fast as stable diffusion on modern nvidia GPU but it might save the day if such harware is not available.

## Building
just run
```sh
go build
```

## Usage

command 
```sh
./stbdif -?
```
gives list of command line arguments

```sh
Usage of ./stbdif:
  -cfgscale float
        CfgScale (default 7)
  -h int
        prefered value depends on model, use power of two (default 512)
  -if string
        input file for img2img operation
  -j string
        run stable diffusion job from json file
  -m string
        model file in ggml format
  -n int
        number of steps (default 10)
  -np string
        default negative prompt if job file not used
  -o string
        output file prefix (default "outsd")
  -od string
        output directory for pictures (default "/tmp/")
  -p string
        default prompt if job file not used
  -r int
        how many repeats of command or  (default 1)
  -schedule string
        DEFAULT, DISCRETE, KARRAS,N_SCHEDULES (default "DEFAULT")
  -seed int
        rng seed (default -1)
  -sm string
        EULER_A,EULER,HEUN,DPM2,DPMPP2S_A,DPMPP2M,DPMPP2Mv2,N_SAMPLE_METHODS (default "EULER")
  -st float
        strength for noising/unnoising img2img. 1=full image desctruction (default 0.75)
  -th int
        number of threads  -1=automatic (default -1)
  -w int
        prefered value depends on model, use power of two (default 512)
```

## job json format

Typical use is to use job file. Parameters on command line override values on job file.
job json file is array of *JobEntry* structs.

Typical json file could look like this (*exampleJob.list*) 

```json
[{
    "outputPrefix":"koiru",
    "inputImage":"",
    "prompt":"superhero dog",
    "negPrompt":"",
    "cfgScale":0.7,
    "width":512,
    "height":512,
    "sampleMethod":"DPM2",
    "sampleSteps":15,
    "strength":0,
    "seed":-1,
    "repeats":2
},
{
    "outputPrefix":"city",
    "inputImage":"",
    "prompt":"cyberpunk cyber city street many robots",
    "negPrompt":"",
    "cfgScale":0.7,
    "width":512,
    "height":512,
    "sampleMethod":"EULER",
    "sampleSteps":35,
    "strength":0,
    "seed":-1,
    "repeats":3
    }
]
```

And it could be runned with command

```sh
./stbdif -m modelfilehere -j exampleJobList.json
```

One way to use this software is to try different kind of options and prompts on command line. Program generates .png files and .json files as result. Intresting picture settings can be collected from json files as one batch job file. And then let run those with high number of repeats.