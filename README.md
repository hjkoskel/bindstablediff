# Bindstablediff

![bindstablediffpic](titlepic.png)

[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) bindings for golang for running stable diffusion on CPU

Goal of this library is provide simple interface for running stable diffusion process on golang programs in situations where no GPU is not available (like cloud service).

This library is under development at the moment. Now this is only thin wrapper around C++ code. Goal is to allow user tap into diffusion process and expand existing algorithm. (zooming, custom tokenizer weighting etc.. tricks)

## Converting models

Original stable-diffusion.cpp site have some guidance how to do that.

Basic use case is to just use float16 and convert .ckpt to ggml binary by conversion script
```shell
	cd convert
    python convert.py sd-v1-4.ckpt --out_type f16
```
## Using library
Basic idea is to include library (and do go mod tidy)
```go
import "github.com/hjkoskel/bindstablediff"
```

Then create *StableDiffusionModel* with function *InitStableDiffusion* for loading model
```go
func InitStableDiffusion(fname string, nThreads int, schedule EnumSchedule) (StableDiffusionModel, error) {
```

Then collect parameters to struct and call txt2img
```go
par := bindstablediff.TextGenPars{
		Prompt:         "cute dog",
		NegativePrompt: "",
		CfgScale:       7,
		Width:          512,
		Height:         512,
		SampleMethod:   bindstablediff.HEUN,
		SampleSteps:    *pSteps,
		Seed:           -1}

resultImg, errGen := engine.Txt2Img(par)
```

## Example dogandcat

Directory ./cmd/dogandcat have minimal example how to use this library.

## Example usage stbdiff

directory ./cmd/stbdiff contains simple commandline utility for batch running.

There are two basic ways to use. Directly from command line or by using json file containing list of jobs.
Its own README.md have more details