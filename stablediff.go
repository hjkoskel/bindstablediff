package bindstablediff

/*
#cgo LDFLAGS: -L. -L${SRCDIR}/src -lm -lstdc++
#cgo CXXFLAGS: -I. -I./ggml/include -pthread -O3 -msse3 -fPIC -m64
#cgo CFLAGS: -march=native
#include "bindstablediff.h"
#include <stdlib.h>
#include <stdio.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"os"
	"runtime"
	"strings"
	"unsafe"
)

type StableDiffusionModel struct {
	sdModel C.StableDiffusionModel
}

type EnumSDLogLevel int

const (
	DEBUG EnumSDLogLevel = 0
	INFO  EnumSDLogLevel = 1
	WARN  EnumSDLogLevel = 2
	ERROR EnumSDLogLevel = 3
)

type EnumRNGType int

const (
	STD_DEFAULT_RNG EnumRNGType = 0
	CUDA_RNG        EnumRNGType = 1
)

type EnumSampleMethod int

const (
	EULER_A          EnumSampleMethod = 0
	EULER            EnumSampleMethod = 1
	HEUN             EnumSampleMethod = 2
	DPM2             EnumSampleMethod = 3
	DPMPP2S_A        EnumSampleMethod = 4
	DPMPP2M          EnumSampleMethod = 5
	DPMPP2Mv2        EnumSampleMethod = 6
	N_SAMPLE_METHODS EnumSampleMethod = 7
)

func ParseSampleMethod(s string) (EnumSampleMethod, error) {
	m := map[string]EnumSampleMethod{
		"EULER_A":          EULER_A,
		"EULER":            EULER,
		"HEUN":             HEUN,
		"DPM2":             DPM2,
		"DPMPP2S_A":        DPMPP2S_A,
		"DPMPP2M":          DPMPP2M,
		"DPMPP2Mv2":        DPMPP2Mv2,
		"N_SAMPLE_METHODS": N_SAMPLE_METHODS,
	}
	result, haz := m[strings.ToUpper(s)]
	if !haz {
		return EULER_A, fmt.Errorf("invalid sample method name %s", s)
	}
	return result, nil
}

type EnumSchedule int

const (
	DEFAULT     EnumSchedule = 0
	DISCRETE    EnumSchedule = 1
	KARRAS      EnumSchedule = 2
	N_SCHEDULES EnumSchedule = 3
)

func ParseSchedule(s string) (EnumSchedule, error) {
	m := map[string]EnumSchedule{
		"DEFAULT":     DEFAULT,
		"DISCRETE":    DISCRETE,
		"KARRAS":      KARRAS,
		"N_SCHEDULES": N_SCHEDULES,
	}

	result, haz := m[strings.ToUpper(s)]
	if !haz {
		return DEFAULT, fmt.Errorf("invalid schedule name %s", s)
	}
	return result, nil
}

func exists(name string) (bool, error) {
	_, err := os.Stat(name)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return false, err
}

func InitStableDiffusion(fname string, nThreads int, schedule EnumSchedule) (StableDiffusionModel, error) {
	if len(fname) == 0 {
		return StableDiffusionModel{}, fmt.Errorf("no model file given")
	}
	fileExists, _ := exists(fname)
	if !fileExists {
		return StableDiffusionModel{}, fmt.Errorf("model file %s not found", fname)
	}

	if nThreads < 1 {
		nThreads = runtime.NumCPU()
	}

	var result StableDiffusionModel

	ret := C.loadStableDiffusion(
		C.CString(fname), //char *sdfilename,
		C.int(nThreads),  //int n_threads,
		C.int(schedule),  //int enumSchedule,
		&result.sdModel)

	if ret != 0 {
		return result, fmt.Errorf("init fail with code %v", ret)
	}
	return result, nil
}

// Lets have parameters as struct.. so it is easier to store to exif etc...
type TextGenPars struct {
	Prompt         string
	NegativePrompt string
	CfgScale       float32 //7 default
	Width          int
	Height         int
	SampleMethod   EnumSampleMethod
	SampleSteps    int
	Strength       float32 //needed for img2img
	Seed           int64
}

func rgb2img(rgb []byte, width int, height int) (image.Image, error) {
	if len(rgb) != width*height*3 {
		return nil, fmt.Errorf("RGB data length %d does not match %d x %d x 3 = %d", len(rgb), width*height*3)
	}
	resultImage := image.NewRGBA(image.Rect(0, 0, width, height))
	pos := 0
	for i := 0; i < len(rgb); i += 3 {
		resultImage.Pix[pos+0] = rgb[i+0]
		resultImage.Pix[pos+1] = rgb[i+1]
		resultImage.Pix[pos+2] = rgb[i+2]
		resultImage.Pix[pos+3] = 255 //A
		pos += 4
	}
	return resultImage, nil
}

func img2rgb(img image.Image) []byte {
	b := img.Bounds()
	m := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(m, m.Bounds(), img, b.Min, draw.Src)

	n := b.Dx() * b.Dy() * 3
	result := make([]byte, n)
	for pixelCounter := 0; pixelCounter < n/3; pixelCounter++ {
		result[pixelCounter*3+0] = m.Pix[pixelCounter*4+0]
		result[pixelCounter*3+1] = m.Pix[pixelCounter*4+1]
		result[pixelCounter*3+2] = m.Pix[pixelCounter*4+2]
	}
	return result
}

func (p *StableDiffusionModel) Txt2Img(parameters TextGenPars) (image.Image, error) {
	rawResult := C.txt2img(&p.sdModel,
		C.CString(parameters.Prompt),
		C.CString(parameters.NegativePrompt),
		C.float(parameters.CfgScale),
		C.int(parameters.Width), C.int(parameters.Height),
		C.int(parameters.SampleMethod),
		C.int(parameters.SampleSteps),
		C.long(parameters.Seed))

	if rawResult == nil {
		return nil, fmt.Errorf("txt2img failed with nil image")
	}
	imagedata := C.GoBytes(unsafe.Pointer(rawResult), C.int(parameters.Width*parameters.Height*3))
	result, convErr := rgb2img(imagedata, parameters.Width, parameters.Height)
	C.free(unsafe.Pointer(rawResult))

	if convErr != nil {
		return nil, convErr
	}

	return result, nil
}

// Img2Img, not yet ready
func (p *StableDiffusionModel) Img2Img(startImage image.Image, parameters TextGenPars) (image.Image, error) {
	if startImage.Bounds().Dx() != parameters.Width || startImage.Bounds().Dy() != parameters.Height {
		return nil, fmt.Errorf("start image dimensions %d x %d do not match image dimensions %d x %d",
			startImage.Bounds().Dx(), startImage.Bounds().Dy(),
			parameters.Width, parameters.Width)
	}

	startImgBytes := img2rgb(startImage)

	/*
		FAILS: TODO FIX
		stable-diffusion.cpp:1407: static void DownSample::asymmetric_pad(ggml_tensor*, const ggml_tensor*, const ggml_tensor*, int, int, void*): Assertion `sizeof(dst->nb[0]) == sizeof(float)' failed.
	*/

	rawResult := C.img2img(&p.sdModel,
		(*C.uchar)(C.CBytes(startImgBytes)),
		C.CString(parameters.Prompt),
		C.CString(parameters.NegativePrompt),
		C.float(parameters.CfgScale),                      //float cfg_scale,
		C.int(parameters.Width), C.int(parameters.Height), //int width,int height,
		C.int(parameters.SampleMethod), //int sampleMethod, //TODO ENUM
		C.int(parameters.SampleSteps),  //int sampleSteps,
		C.float(parameters.Strength),
		C.long(parameters.Seed)) //int64_t seed)

	if rawResult == nil {
		return nil, fmt.Errorf("txt2img failed with nil image")
	}
	imagedata := C.GoBytes(unsafe.Pointer(rawResult), C.int(parameters.Width*parameters.Height*3))
	return rgb2img(imagedata, parameters.Width, parameters.Height)
}

func SavePng(fname string, img image.Image) error {
	f, err := os.Create(fname)
	if err != nil {
		return fmt.Errorf("create image fail %v", err)
	}
	if err := png.Encode(f, img); err != nil {
		return fmt.Errorf("error writing %s failed err=%v", fname, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("error writing %s failed err=%v", fname, err)
	}
	return nil
}
