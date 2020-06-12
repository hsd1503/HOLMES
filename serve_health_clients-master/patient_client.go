package main

import (
	"fmt"
	"os"
	"io/ioutil"
	"runtime"
	//"net"
	"net/http"
	"time"
	"math"
	"math/rand"
	"log"
)

var Maximum_backoff = 32000 //in millisecond

func on_disconnect(client *http.Client, url string, backoff_counter int, ch chan<- string){
	backoff_counter++
	increasing_backoff := math.Pow(2, float64(backoff_counter))
	random_number_milliseconds := rand.Intn(10000)
	final_backoff := math.Min((increasing_backoff+float64(random_number_milliseconds)), float64(Maximum_backoff))
	// fmt.Printf("backing off for the %d time: %.2f ", backoff_counter, final_backoff)
	time.Sleep(time.Duration(final_backoff) * time.Millisecond)
	// fmt.Println("finish sleeping : %.2f, now call request %s", final_backoff, url)
	MakeRequest(client, url, backoff_counter, ch)
}

func MakeRequest(client *http.Client, url string, backoff_counter int, ch chan<- string) {
	// fmt.Println("start request ", url)
	start := time.Now()
	resp, err := client.Get(url)
	secs := time.Since(start).Seconds()
	if err != nil || resp == nil {
		fmt.Println("handle error get")
		log.Fatalf("on_disconnect: ", err)
		//on_disconnect(client, url, backoff_counter, ch)
	}

	if resp != nil {
		defer resp.Body.Close()
		body, errRead := ioutil.ReadAll(resp.Body)
		if errRead != nil {
			fmt.Println("handle error read response body")
			log.Fatalf("Error in reading: ", errRead)
		}
		ch <- fmt.Sprintf("%.2f elapsed with response length: %s %s", secs, body, url)
	}
}

func main() {
	runtime.GOMAXPROCS(5)
	patient_name := os.Args[1]
	ip := os.Args[2]
	fmt.Println(patient_name,ip)

	// client := &http.Client{}
	// ch := make(chan string)
	address := "http://"+ip+"/hospital?patient_name="+patient_name+"&value=0.0&vtype=ECG"
	fmt.Println(address)
	totalRequest := 3751
	tr := &http.Transport{
		// DialContext:(&net.Dialer{
  //           Timeout:   300 * time.Second,
  //       }).DialContext,
		// TLSHandshakeTimeout:   300 * time.Second,
		// MaxIdleConns:100,
		MaxIdleConnsPerHost: 30,
		MaxConnsPerHost: 31,
		// IdleConnTimeout:300 * time.Second,
	}
	client := &http.Client{Transport: tr}
	//Timeout: 300 * time.Second}

	start := time.Now()
	ch := make(chan string)
	for i := 0; i <= totalRequest; i++ {
		// wait for 8 milliseconds to simulate the patient
		// incoming data
		time.Sleep(8 * time.Millisecond)
		// This how actual client will send the result
		// go MakeRequest("http://127.0.0.1:5000/hospital?patient_name=Adam&value=0.0&vtype=ECG", ch)
		// This is how profiling result is send
		// fmt.Printf("client %s alive : loop: %d ", *patientId, i)
		go MakeRequest(client, address, 0, ch)
	}
	for i := 0; i <= totalRequest; i++ {
		fmt.Println(<-ch)
	}
	fmt.Printf("client finished %.2fs elapsed\n", time.Since(start).Seconds())
	// sleep 1 minute to make sure all previous request and socket file descriptor are closed
	// fmt.Println("start sleeping to make sure all socket killed")
	// time.Sleep(time.Minute * 1)
}
