package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"time"
)

func MakeRequest(url string, ch chan<- string) {
	start := time.Now()
	resp, _ := http.Get(url)
	secs := time.Since(start).Seconds()
	body, _ := ioutil.ReadAll(resp.Body)
	ch <- fmt.Sprintf("%.2f elapsed with response length: %s %s", secs, body, url)
}
func main() {
	start := time.Now()
	patient_name := os.Args[1]
	time_ms, err := strconv.ParseFloat(os.Args[2], 64)
	if err != nil {
		time_ms = 20.0
	}
	fmt.Println(patient_name)
	fmt.Print(time_ms)
	ch := make(chan string)
	for i := 0; i <= 100; i++ {
		// wait for 8 milliseconds to simulate the patient
		// incoming data
		time.Sleep(time.Duration(time_ms) * time.Millisecond)
		// This how actual client will send the result
		go MakeRequest("http://127.0.0.1:8000/profileEnsemble?patient_name="+
			patient_name+"&value=0.0&vtype=ECG", ch)
		// This is how profiling result is send
		//go MakeRequest("http://127.0.0.1:8000/RayServeProfile/hospital", ch)
	}
	for i := 0; i <= 100; i++ {
		fmt.Println(<-ch)
	}
	fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}
