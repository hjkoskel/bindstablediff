/*
simple program for generating cat and dog pictures

Used as simple example on readme
*/
package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/hjkoskel/bindstablediff"
)

func SavePng(fname string, img image.Image) error {
	f, err := os.Create(fname)
	if err != nil {
		fmt.Printf("create image fail %v", err)
	}
	if err := png.Encode(f, img); err != nil {
		return fmt.Errorf("error writing %s failed err=%v", fname, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("error writing %s failed err=%v", fname, err)
	}
	return nil
}

func main() {
	pModel := flag.String("m", "", "model filename")
	pSteps := flag.Int("n", 10, "number of steps use lower if slow cpu")
	pNumberOfThreads := flag.Int("t", -1, "number of threads, if -1 then autodetect number of cores")
	flag.Parse()

	if len(*pModel) == 0 {
		fmt.Printf("please provide model filename with -m flag\n")
		return
	}

	engine, errInit := bindstablediff.InitStableDiffusion(*pModel, *pNumberOfThreads, bindstablediff.KARRAS)
	if errInit != nil {
		fmt.Printf("error initialized %v\n", errInit)
		return
	}

	par := bindstablediff.TextGenPars{
		Prompt:         "cute dog",
		NegativePrompt: "",
		CfgScale:       7,
		Width:          512,
		Height:         512,
		SampleMethod:   bindstablediff.EULER,
		SampleSteps:    *pSteps,
		Seed:           -1}

	resultImg, errGen := engine.Txt2Img(par)
	if errGen != nil {
		fmt.Printf("error generating %s\n", errGen.Error())
		return
	}
	errSave := SavePng("koira1.png", resultImg)
	if errSave != nil {
		fmt.Printf("error saving %s\n", errSave.Error())
		return
	}

	//Lets just change prompt
	par.Prompt = "cute cat"
	resultImg, errGen = engine.Txt2Img(par)
	if errGen != nil {
		fmt.Printf("error generating %s\n", errGen.Error())
		return
	}
	errSave = SavePng("kissa1.png", resultImg)
	if errSave != nil {
		fmt.Printf("error saving %s\n", errSave.Error())
		return
	}
}
