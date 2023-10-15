/*
Simple command line stable diffusion program
*/
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"
	"path"
	"strings"
	"time"

	"github.com/hjkoskel/bindstablediff"
)

func SavePng(img image.Image, fname string) error {
	f, err := os.Create(fname)
	if err != nil {
		fmt.Printf("create image fail %v", err)
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return fmt.Errorf("error writing %s failed err=%v", fname, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("error writing %s failed err=%v", fname, err)
	}
	return nil
}

func LoadPng(fname string) (image.Image, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	return img, err
}

func Exists(name string) (bool, error) {
	_, err := os.Stat(name)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return false, err
}

// Checks the next available png name
func CreateOutputFileName(dirname string, seedNumber int64, prefix string) (string, error) {
	//Is there need for better name?
	fname := path.Join(dirname, fmt.Sprintf("%v_%v.png", prefix, seedNumber))
	alreadyHaveFile, errCheckExist := Exists(fname)
	if errCheckExist != nil {
		return "", fmt.Errorf("error checking file %s err=%s", fname, errCheckExist.Error())
	}
	n := 0
	for alreadyHaveFile {
		n++
		fname = path.Join(dirname, fmt.Sprintf("%v_%v_%v.png", prefix, seedNumber, n))
		alreadyHaveFile, errCheckExist = Exists(fname)
		if errCheckExist != nil {
			return "", fmt.Errorf("error checking file %s err=%s", fname, errCheckExist.Error())
		}
	}
	return fname, nil
}

func main() {
	pModelFile := flag.String("m", "", "model file in ggml format")
	pNumberOfThreads := flag.Int("th", -1, "number of threads  -1=automatic")
	pRepeat := flag.Int("r", 1, "how many repeats of command or ")
	pScheduleString := flag.String("schedule", "DEFAULT", "DEFAULT, DISCRETE, KARRAS,N_SCHEDULES")
	//other parameters
	pOutputDir := flag.String("od", "/tmp/", "output directory for pictures")
	pPrompt := flag.String("p", "", "default prompt if job file not used")
	pNegPrompt := flag.String("np", "", "default negative prompt if job file not used")

	//TODO TBD pInputImage2Image := flag.String("if", "", "input file for img2img operation")
	pJobFile := flag.String("j", "", "run stable diffusion job from json file")

	pOutputPrefix := flag.String("o", "outsd", "output file prefix")

	//parameters directly for render, overrides what job say
	pCfgScale := flag.Float64("cfgscale", 7.0, "CfgScale")
	pWidth := flag.Int("w", 512, "prefered value depends on model, use power of two")
	pHeight := flag.Int("h", 512, "prefered value depends on model, use power of two")
	pSampleMethodString := flag.String("sm", "EULER", "EULER_A,EULER,HEUN,DPM2,DPMPP2S_A,DPMPP2M,DPMPP2Mv2,N_SAMPLE_METHODS")
	pSampleSteps := flag.Int("n", 10, "number of steps") //TODO sample size? vs number of steps?
	pStrength := flag.Float64("st", 0.75, "strength for noising/unnoising img2img. 1=full image desctruction")
	pSeed := flag.Int64("seed", -1, "rng seed") // non -1,
	flag.Parse()

	flagAvailMap := make(map[string]bool)

	flag.Visit(func(f *flag.Flag) {
		fmt.Printf("name=%s\n", f.Name)
		switch f.Name {
		//case
		//flagAvailMap["OutputPrefix"]=
		case "if":
			flagAvailMap["InputImage"] = true
		case "p":
			flagAvailMap["Prompt"] = true
		case "np":
			flagAvailMap["NegPrompt"] = true
		//flagAvailMap["Words"]=
		case "cfgscale":
			flagAvailMap["CfgScale"] = true
		case "w":
			flagAvailMap["Width"] = true
		case "h":
			flagAvailMap["Height"] = true
		case "sm":
			flagAvailMap["SampleMethod"] = true
		case "n":
			flagAvailMap["SampleSteps"] = true
		case "st":
			flagAvailMap["Strength"] = true
		case "seed":
			flagAvailMap["Seed"] = true
		case "o":
			flagAvailMap["OutputPrefix"] = true
		}
	})

	//Get default settings
	var rawJobFile []byte
	if len(*pJobFile) == 0 {
		rawJobFile = nil
	} else {
		var errRead error
		rawJobFile, errRead = os.ReadFile(*pJobFile)
		if errRead != nil {
			fmt.Printf("error reading job file %s err=%s\n", *pJobFile, errRead.Error())
			os.Exit(-1)
		}
	}

	jobArray, parseErr := ParseJobs(rawJobFile, JobEntry{
		OutputPrefix: *pOutputPrefix, //Comes from prompt or defined by job
		//TBD InputImage:   *pInputImage2Image, //img2img mode

		Prompt:    *pPrompt,
		NegPrompt: *pNegPrompt,

		CfgScale: *pCfgScale,
		Width:    *pWidth,
		Height:   *pHeight,

		SampleMethod: *pSampleMethodString,
		SampleSteps:  *pSampleSteps,

		Strength: *pStrength,
		Seed:     *pSeed,
	}, flagAvailMap)

	if parseErr != nil {
		fmt.Printf("parsing job file %s err=%s\n", *pJobFile, parseErr)
	}
	//PRE check
	checkOk := true
	for jobIndex, job := range jobArray {
		//fmt.Printf("job%v:  %#v\n", jobIndex, job)
		errSanity := job.SanityCheck()
		if errSanity != nil {
			fmt.Printf("invalid job%v: %s\n", jobIndex, errSanity.Error())
			checkOk = false
		}
	}
	if !checkOk {
		os.Exit(-1)
	}

	chosenSchedule, scheduleErr := bindstablediff.ParseSchedule(*pScheduleString)
	if scheduleErr != nil {
		fmt.Printf("invalid schedule format %s", scheduleErr.Error())
		os.Exit(-1)
	}

	engine, errInit := bindstablediff.InitStableDiffusion(*pModelFile, *pNumberOfThreads, chosenSchedule)
	if errInit != nil {
		fmt.Printf("error initializing stable diffusion %s\n", errInit.Error())
		os.Exit(-1)
	}

	for repeatCount := 0; repeatCount < *pRepeat || *pRepeat < 0; repeatCount++ {
		for jobIndex, job := range jobArray {
			//fmt.Printf("job have %v repeats\n", job.Repeats)
			for jobRepeatCounter := 0; jobRepeatCounter < job.Repeats; jobRepeatCounter++ {
				parameters, errParameters := job.ToTextGenPars()
				if errParameters != nil {
					fmt.Printf("job%,  %#v have invalid parameters %s\n", jobIndex, job, errParameters.Error())
					os.Exit(-1)
				}
				var genError error
				var generatedPic image.Image

				tGenStart := time.Now()
				if len(job.InputImage) == 0 {
					generatedPic, genError = engine.Txt2Img(parameters)
				} else {
					if parameters.Strength <= 0 {
						fmt.Printf("ERR: strength is %s\n", &parameters.Strength)
					}

					startImage, errLoadImage := LoadPng(job.InputImage)
					if errLoadImage != nil {
						fmt.Printf("error loading %s  err=%s\n", job.InputImage, errLoadImage.Error())
						os.Exit(-1)
					}
					generatedPic, genError = engine.Img2Img(startImage, parameters)
				}

				if genError != nil {
					fmt.Printf("Job%v %#v failed gen error=%s\n", jobIndex, job, genError.Error())
					os.Exit(-1)
				}

				tGenEnd := time.Now()
				fmt.Printf("\n-------job%v generated, saving... ----\n", jobIndex)
				outputFileName, nameErr := CreateOutputFileName(*pOutputDir, parameters.Seed, job.OutputPrefix)
				if nameErr != nil {
					fmt.Printf("file naming error %s\n", nameErr.Error())
					os.Exit(-1)
				}
				saveErr := SavePng(generatedPic, outputFileName)
				if saveErr != nil {
					fmt.Printf("ERROR SAVING %s\n", saveErr.Error())
					os.Exit(-1)
				}

				completedInfo := JobCompletedInfo{
					Filename:    outputFileName,
					RunDuration: tGenEnd.Sub(tGenStart).Milliseconds(),
					Job:         job,
				}
				completedInfo.Job.Seed = parameters.Seed
				infoBytes, _ := json.MarshalIndent(completedInfo, "", " ")
				infoWriteErr := os.WriteFile(strings.Replace(outputFileName, ".png", ".json", 1), infoBytes, 0666)
				if infoWriteErr != nil {
					fmt.Printf("info err %v\n", infoWriteErr.Error())
					os.Exit(-1)
				}
				fmt.Printf("generated %s in %s\n", outputFileName, tGenEnd.Sub(tGenStart))
			}
		}
	}
}
