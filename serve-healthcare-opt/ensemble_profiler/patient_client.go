package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
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
	fmt.Println(patient_name)
	ch := make(chan string)
	for i := 0; i <= 3800; i++ {
		// wait for 8 milliseconds to simulate the patient
		// incoming data
		time.Sleep(8 * time.Millisecond)
		// This how actual client will send the result
		go MakeRequest("http://127.0.0.1:8000/hospital?patient_name="+
			patient_name+"&value=0.0&vtype=ECG", ch)
		// This is how profiling result is send
		//go MakeRequest("http://127.0.0.1:8000/RayServeProfile/hospital", ch)
	}
	for i := 0; i <= 3800; i++ {
		fmt.Println(<-ch)
	}
	fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}
