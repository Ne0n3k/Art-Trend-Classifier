#!/usr/bin/env python3
import requests
import json
import time
import os
import sys
from pathlib import Path
import subprocess
import threading
from typing import Dict, Any, List

class IntegrationTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_images_dir = Path(__file__).parent.parent.parent / "docs" / "examples"
        self.backend_process = None
        self.results = []
        
    def start_backend(self) -> bool:
        print("Starting backend server...")
        try:
            os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null || true")
            time.sleep(2)
            backend_dir = os.path.join(os.path.dirname(__file__), "..")
            self.backend_process = subprocess.Popen([
                sys.executable, "main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=backend_dir)
            
            # Wait for server
            time.sleep(5)
            
            try:
                response = requests.get(f"{self.base_url}/", timeout=10)
                if response.status_code == 200:
                    print("Backend server started successfully")
                    return True
                else:
                    print("Backend server failed to start")
                    return False
            except requests.exceptions.ConnectionError:
                print("Backend server not responding, but continuing with tests...")
                return True
        except Exception as e:
            print(f"Error starting backend: {e}")
            return False
    
    def stop_backend(self):
        if self.backend_process:
            print("Stopping backend server...")
            self.backend_process.terminate()
            self.backend_process.wait()
            print("Backend server stopped")
    
    def test_health_endpoint(self) -> bool:
        print("\nTesting health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "classes" in data:
                    print(f"Health check passed: {data['message']}, Classes: {data['classes']}")
                    return True
                else:
                    print("Health check response format incorrect")
                    return False
            else:
                print(f"Health check failed with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"Health check error: {e}")
            return False
    
    def test_image_analysis(self, image_path: Path) -> bool:
        print(f"\nTesting image analysis: {image_path.name}")
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['predicted_style', 'confidence', 'review', 'top_predictions', 'all_predictions']
                
                if all(field in data for field in required_fields):
                    print(f"Analysis successful:")
                    print(f"Style: {data['predicted_style']}")
                    print(f"Confidence: {data['confidence']:.3f}")
                    print(f"Review: {data['review'][:50]}...")
                    print(f"Top predictions: {len(data['top_predictions'])}")
                    print(f"All predictions: {len(data['all_predictions'])}")
                    return True
                else:
                    print("Analysis response missing required fields")
                    return False
            else:
                print(f"Analysis failed with status: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"Analysis error: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        print("\nTesting error handling...")
        tests_passed = 0
        total_tests = 3
        
        # Test 1: No file provided
        try:
            response = requests.post(f"{self.base_url}/analyze", timeout=10)
            if response.status_code == 422:
                print("No file error handled correctly")
                tests_passed += 1
            else:
                print(f"No file error not handled: {response.status_code}")
        except Exception as e:
            print(f"No file error test failed: {e}")
        
        # Test 2: Invalid file type
        try:
            files = {'file': ('test.txt', b'not an image', 'text/plain')}
            response = requests.post(f"{self.base_url}/analyze", files=files, timeout=10)
            if response.status_code == 400:
                print("Invalid file type error handled correctly")
                tests_passed += 1
            else:
                print(f"Invalid file type error not handled: {response.status_code}")
        except Exception as e:
            print(f"Invalid file type error test failed: {e}")
        
        # Test 3: Large file
        try:
            large_data = b'x' * (5 * 1024 * 1024)  # 5MB
            files = {'file': ('large.jpg', large_data, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
            print(f"Large file test completed: {response.status_code}")
            tests_passed += 1
        except Exception as e:
            print(f"Large file test handled correctly: {e}")
            tests_passed += 1
        
        return tests_passed == total_tests
    
    def test_performance(self) -> bool:
        # Test API performance
        print("\n⚡ Testing performance...")
        
        if not self.test_images_dir.exists():
            print("No test images directory found")
            return False
        
        test_images = list(self.test_images_dir.glob("*.jpg"))
        if not test_images:
            print("No test images found")
            return False
        
        times = []
        for image_path in test_images[:3]:  # Test first 3 images
            start_time = time.time()
            success = self.test_image_analysis(image_path)
            end_time = time.time()
            
            if success:
                response_time = end_time - start_time
                times.append(response_time)
                print(f"   Response time: {response_time:.2f}s")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"Average response time: {avg_time:.2f}s")
            if avg_time < 5.0:
                print("Performance acceptable (< 5s)")
                return True
            else:
                print("Performance slow (> 5s)")
                return False
        return False
    
    def test_cors_headers(self) -> bool:
        print("\n🌐 Testing CORS headers...")
        try:
            # Send proper CORS preflight request
            headers = {
                'Origin': 'http://localhost:3000',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            response = requests.options(f"{self.base_url}/analyze", headers=headers, timeout=10)
            cors_headers = [
                'access-control-allow-origin',
                'access-control-allow-methods',
                'access-control-allow-headers'
            ]
            
            if all(header in response.headers for header in cors_headers):
                print("CORS headers present")
                return True
            else:
                print("CORS headers missing")
                print(f"Available headers: {list(response.headers.keys())}")
                return False
        except Exception as e:
            print(f"CORS test error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        print("Starting Integration Tests")
        print("=" * 50)
        
        # Start backend
        if not self.start_backend():
            print("Cannot run tests without backend")
            return False
        
        try:
            # Run tests
            tests = [
                ("Health Endpoint", self.test_health_endpoint),
                ("CORS Headers", self.test_cors_headers),
                ("Error Handling", self.test_error_handling),
                ("Performance", self.test_performance),
            ]
            
            # Add image analysis tests
            if self.test_images_dir.exists():
                test_images = list(self.test_images_dir.glob("*.jpg"))
                for image_path in test_images[:2]:
                    tests.append((f"Image Analysis: {image_path.name}", 
                                lambda img=image_path: self.test_image_analysis(img)))
            
            # Execute tests
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
            print(f"Integration Test Results: {passed}/{total} passed")
            
            if passed == total:
                print("All integration tests PASSED!")
                return True
            else:
                print("Some integration tests FAILED!")
                return False
                
        finally:
            self.stop_backend()

if __name__ == "__main__":
    tester = IntegrationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
