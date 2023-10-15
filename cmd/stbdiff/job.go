package main

import (
	"encoding/json"
	"fmt"
	"math/rand"

	"github.com/hjkoskel/bindstablediff"
)

type JobEntry struct {
	OutputPrefix string `json:"outputPrefix,omitempty"` //For image generated
	InputImage   string `json:"inputImage,omitempty"`   //img2img mode

	Prompt    string `json:"prompt,omitempty"` //alternative to word map with weight coefficents (does CLIP )
	NegPrompt string `json:"negPrompt,omitempty"`

	CfgScale float64 `json:"cfgScale,omitempty"`
	Width    int     `json:"width,omitempty"`
	Height   int     `json:"height,omitempty"`

	SampleMethod string `json:"sampleMethod,omitempty"`
	SampleSteps  int    `json:"sampleSteps,omitempty"`

	Strength float64 `json:"strength,omitempty"`
	Seed     int64   `json:"seed,omitempty"`

	Repeats int `json:"repeats,omitempty"` //How many repeats
}

// Created after run and saved as .json file. Can be used for creating  new job entries
type JobCompletedInfo struct {
	Filename    string   `json:"filename,omitempty"`
	RunDuration int64    `json:"runDuration,omitempty"`
	Job         JobEntry `json:"job,omitempty"`
}

func (p *JobEntry) ToTextGenPars() (bindstablediff.TextGenPars, error) {

	seed := p.Seed
	if seed < 0 {
		seed = rand.Int63()
	}

	sampleMethod, sampleMethodErr := bindstablediff.ParseSampleMethod(p.SampleMethod)
	if sampleMethodErr != nil {
		return bindstablediff.TextGenPars{}, fmt.Errorf("invalid sample method %s", sampleMethodErr.Error())
	}

	return bindstablediff.TextGenPars{
		Prompt:         p.Prompt,
		NegativePrompt: p.NegPrompt,
		CfgScale:       float32(p.CfgScale), //7 default
		Width:          p.Width,
		Height:         p.Height,
		SampleMethod:   sampleMethod,
		SampleSteps:    p.SampleSteps,       //TODO sample size? vs number of steps?
		Strength:       float32(p.Strength), //needed for img2img
		Seed:           seed}, nil
}

func (p *JobEntry) SanityCheck() error {
	if len(p.OutputPrefix) == 0 {
		return fmt.Errorf("output prefix requred")
	}
	_, sampleMethodErr := bindstablediff.ParseSampleMethod(p.SampleMethod)
	if sampleMethodErr != nil {
		return fmt.Errorf("invalid sample method %s", sampleMethodErr.Error())
	}
	//TODO range checks etc... TODO POWER OF TWO PICTURE DIMENSIONS!
	if len(p.Prompt) == 0 && len(p.NegPrompt) == 0 && len(p.InputImage) == 0 {
		return fmt.Errorf("prompt or some input data required")
	}
	return nil
}

// ParseJobs parses JSON array of jobs and handles default settings thing
func ParseJobs(raw []byte, defaultValues JobEntry, overridedValues map[string]bool) ([]JobEntry, error) {
	result := []JobEntry{}
	fmt.Printf("going to parse jobs with default %#v\n\n\n", defaultValues)
	if raw == nil { //No job file, just defaulst and overrided values
		result = []JobEntry{defaultValues}
	} else {
		errParse := json.Unmarshal(raw, &result)
		if errParse != nil {
			return nil, errParse
		}
	}

	//Lets override values that are defined with defaults
	//Does some values have option to override command line arguments?
	for name, isSet := range overridedValues {
		if !isSet {
			overridedValues[name] = false
			continue
		}
	}
	for i, _ := range result { //Loop thru all entries
		if result[i].Repeats < 1 {
			result[i].Repeats = 1
		}
		if len(result[i].OutputPrefix) == 0 { //replace only empty
			result[i].OutputPrefix = defaultValues.OutputPrefix
		}

		is := overridedValues["InputImage"]
		if is {
			result[i].InputImage = defaultValues.InputImage
		}
		is = overridedValues["Promptcase"]
		if len(result[i].Prompt) == 0 {
			result[i].Prompt = defaultValues.Prompt
		}

		is = overridedValues["NegPrompt"]
		if len(result[i].NegPrompt) == 0 {
			result[i].NegPrompt = defaultValues.NegPrompt
		}

		is = overridedValues["CfgScale"]
		if is {
			result[i].CfgScale = defaultValues.CfgScale
		}
		is = overridedValues["Width"]
		if is {
			result[i].Width = defaultValues.Width //If defined, override
		}
		is = overridedValues["Height"]
		if is {
			result[i].Height = defaultValues.Height //If defined, override
		}
		is = overridedValues["SampleMethod"]
		if is {
			result[i].SampleMethod = defaultValues.SampleMethod
		}
		is = overridedValues["SampleSteps"]
		if is {
			result[i].SampleSteps = defaultValues.SampleSteps
		}
		is = overridedValues["Strength"]
		if is {
			result[i].Strength = defaultValues.Strength
		}
		is = overridedValues["Seed"]
		if is {
			result[i].Seed = defaultValues.Seed
		}
	}

	return result, nil
}

/*
func ParseJobEntry(defaultValues JobEntry, raw interface{}) (JobEntry, error) {
	//In json... allow constants enums (weak point in golang normal json marshal unmarshal)
	//var raw interface{}

	err := json.Unmarshal(b, &f)

}
*/
