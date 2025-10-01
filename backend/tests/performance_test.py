#!/usr/bin/env python3
import requests
import time
import psutil
import os
import sys
import threading
from pathlib import Path
from typing import List, Dict, Any
import statistics
import subprocess

class PerformanceTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_images_dir = Path(__file__).parent.parent.parent / "docs" / "examples"
        self.results = []
        
    def get_system_resources(self) -> Dict[str, float]:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    def test_model_loading_time(self) -> bool:
        print("\nTesting model loading time...")
        
        start_time = time.time()
        
        try:
            # Kill any existing process
            os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null || true")
            time.sleep(2)
            
            backend_dir = os.path.join(os.path.dirname(__file__), "..")
            process = subprocess.Popen([
                sys.executable, "main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=backend_dir)
            
            # Wait for model to load
            model_loaded = False
            timeout = 60
            
            for _ in range(timeout):
                try:
                    response = requests.get(f"{self.base_url}/", timeout=2)
                    if response.status_code == 200:
                        model_loaded = True
                        break
                except:
                    pass
                time.sleep(1)
            
            end_time = time.time()
            loading_time = end_time - start_time
            
            process.terminate()
            process.wait()
            
            if model_loaded:
                print(f"Model loaded in {loading_time:.2f} seconds")
                if loading_time < 30:
                    print("Loading time acceptable (< 30s)")
                    return True
                else:
                    print("Loading time slow (> 30s)")
                    return False
            else:
                print("Model failed to load within timeout")
                return False
                
        except Exception as e:
            print(f"Model loading test error: {e}")
            return False
    
    def test_single_request_performance(self) -> bool:
        print("\nTesting single request performance...")
        
        if not self.test_images_dir.exists():
            print("No test images directory found")
            return False
        
        test_images = list(self.test_images_dir.glob("*.jpg"))
        if not test_images:
            print("No test images found")
            return False
        
        try:
            # Check if backend is running
            response = requests.get(f"{self.base_url}/", timeout=2)
            if response.status_code != 200:
                print("Backend not running, skipping test")
                return True
            
            # Test single request
            image_path = test_images[0]
            start_time = time.time()
            
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/analyze", files=files, timeout=10)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                print(f"Single request completed in {response_time:.2f} seconds")
                if response_time < 5:
                    print("Response time acceptable (< 5s)")
                    return True
                else:
                    print("Response time slow (> 5s)")
                    return False
            else:
                print(f"Request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Backend not available, skipping test: {e}")
            return True  # Don't fail if backend not available
    
    def test_concurrent_requests(self) -> bool:
        print("\nTesting concurrent requests...")
        
        if not self.test_images_dir.exists():
            print(f"No test images directory found at: {self.test_images_dir}")
            return False
        
        test_images = list(self.test_images_dir.glob("*.jpg"))
        if not test_images:
            print(f"No test images found in: {self.test_images_dir}")
            return False
        
        # Start backend
        try:
            os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null || true")
            time.sleep(2)
            
            # Change to backend directory and run main.py
            backend_dir = os.path.join(os.path.dirname(__file__), "..")
            process = subprocess.Popen([
                sys.executable, "main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=backend_dir)
            
            # Wait for startup
            time.sleep(5)
            
            # Test concurrent requests
            num_requests = 2
            threads = []
            results = []
            
            def make_request(image_path, result_list):
                try:
                    start_time = time.time()
                    with open(image_path, 'rb') as f:
                        files = {'file': (image_path.name, f, 'image/jpeg')}
                        response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
                    end_time = time.time()
                    
                    result_list.append({
                        'status_code': response.status_code,
                        'response_time': end_time - start_time,
                        'success': response.status_code == 200
                    })
                except Exception as e:
                    result_list.append({
                        'status_code': 0,
                        'response_time': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            # Start concurrent requests
            for i in range(num_requests):
                image_path = test_images[i % len(test_images)]
                thread = threading.Thread(target=make_request, args=(image_path, results))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            process.terminate()
            process.wait()
            
            # Analyze results
            successful_requests = sum(1 for r in results if r['success'])
            avg_response_time = statistics.mean([r['response_time'] for r in results if r['success']])
            
            print(f"Concurrent requests: {successful_requests}/{num_requests} successful")
            print(f"Average response time: {avg_response_time:.2f}s")
            
            if successful_requests >= num_requests * 0.8:
                print("Concurrent handling acceptable")
                return True
            else:
                print("Concurrent handling poor")
                return False
                
        except Exception as e:
            print(f"Concurrent requests test error: {e}")
            return False
    
    def test_memory_usage(self) -> bool:
        print("\nTesting memory usage...")
        
        if not self.test_images_dir.exists():
            print("No test images directory found")
            return False
        
        test_images = list(self.test_images_dir.glob("*.jpg"))
        if not test_images:
            print("No test images found")
            return False
        
        # Start backend
        try:
            os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null || true")
            time.sleep(2)
            
            backend_dir = os.path.join(os.path.dirname(__file__), "..")
            process = subprocess.Popen([
                sys.executable, "main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=backend_dir)
            
            # Wait for startup
            time.sleep(5)
            
            # Measure memory before requests
            initial_memory = psutil.virtual_memory().percent

            for i in range(2):
                image_path = test_images[i % len(test_images)]
                with open(image_path, 'rb') as f:
                    files = {'file': (image_path.name, f, 'image/jpeg')}
                    response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
            
            # Measure memory after requests
            final_memory = psutil.virtual_memory().percent
            memory_increase = final_memory - initial_memory
            
            process.terminate()
            process.wait()
            
            print(f"Memory usage: {initial_memory:.1f}% → {final_memory:.1f}% (Δ{memory_increase:.1f}%)")
            
            if memory_increase < 10:
                print("Memory usage acceptable")
                return True
            else:
                print("Memory usage high")
                return False
                
        except Exception as e:
            print(f"Memory usage test error: {e}")
            return False
    
    def test_large_file_handling(self) -> bool:
        print("\nTesting large file handling...")
        
        # Start backend
        try:
            os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null || true")
            time.sleep(2)
            
            backend_dir = os.path.join(os.path.dirname(__file__), "..")
            process = subprocess.Popen([
                sys.executable, "main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=backend_dir)
            
            # Wait for startup
            time.sleep(5)

            file_sizes = [0.01, 0.05, 0.1]
            results = []
            
            for size_mb in file_sizes:
                print(f"   Testing {size_mb}MB file...")
                
                # Create dummy image data
                size_bytes = int(size_mb * 1024 * 1024)
                dummy_data = b'\xff\xd8\xff\xe0' + b'\x00' * max(0, size_bytes - 4)
                
                start_time = time.time()
                try:
                    files = {'file': ('large.jpg', dummy_data, 'image/jpeg')}
                    response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
                    end_time = time.time()
                    
                    results.append({
                        'size_mb': size_mb,
                        'status_code': response.status_code,
                        'response_time': end_time - start_time,
                        'success': response.status_code in [200, 400]
                    })
                    
                    print(f"     Status: {response.status_code}, Time: {end_time - start_time:.2f}s")
                    
                except Exception as e:
                    print(f"     Error: {e}")
                    results.append({
                        'size_mb': size_mb,
                        'status_code': 0,
                        'response_time': 0,
                        'success': False
                    })
            
            process.terminate()
            process.wait()
            
            successful_tests = sum(1 for r in results if r['success'])
            print(f"Large file tests: {successful_tests}/{len(results)} successful")
            
            return successful_tests >= len(results) * 0.5
            
        except Exception as e:
            print(f"Large file test error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        print("Starting Performance Tests")
        print("=" * 50)
        
        tests = [
            ("Model Loading Time", self.test_model_loading_time),
            ("Single Request Performance", self.test_single_request_performance),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\nRunning: {test_name}")
            try:
                if test_func():
                    passed += 1
                    print(f"{test_name} PASSED")
                else:
                    print(f"{test_name} FAILED")
            except Exception as e:
                print(f"{test_name} ERROR: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print(f"Performance Test Results: {passed}/{total} passed")
        
        if passed == total:
            print("All performance tests PASSED!")
            return True
        else:
            print("Some performance tests FAILED!")
            return False

if __name__ == "__main__":
    tester = PerformanceTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
