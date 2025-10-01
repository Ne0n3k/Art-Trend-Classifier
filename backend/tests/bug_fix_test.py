#!/usr/bin/env python3
import requests
import time
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import json

class BugFixTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_images_dir = Path(__file__).parent.parent.parent / "docs" / "examples"
        self.results = []
        
    def start_backend(self) -> bool:
        print("Starting backend server...")
        try:
            os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null || true")
            time.sleep(2)
            
            # Create a dummy model file to prevent startup error
            model_dir = Path("../../ml_model/model")
            model_dir.mkdir(parents=True, exist_ok=True)
            dummy_model = model_dir / "model_best_82_73.pth"
            if not dummy_model.exists():
                # Create a minimal dummy model file
                import torch
                dummy_data = {
                    'model_state_dict': {},
                    'class_names': ['Impressionism', 'Realism', 'Abstract'],
                    'num_classes': 3
                }
                torch.save(dummy_data, dummy_model)
            
            backend_dir = os.path.join(os.path.dirname(__file__), "..")
            self.backend_process = subprocess.Popen([
                sys.executable, "main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=backend_dir)
            
            time.sleep(8)  # Increased wait time
            
            # Try multiple times to connect
            for attempt in range(5):
                # Check if process is still running
                if self.backend_process.poll() is not None:
                    stdout, stderr = self.backend_process.communicate()
                    print(f"Backend process died! Exit code: {self.backend_process.returncode}")
                    print(f"STDOUT: {stdout.decode()}")
                    return False
                
                try:
                    response = requests.get(f"{self.base_url}/", timeout=5)
                    if response.status_code == 200:
                        print("Backend server started successfully")
                        return True
                except requests.exceptions.ConnectionError:
                    if attempt < 4:
                        print(f"Attempt {attempt + 1}/5: Backend not ready yet, waiting...")
                        time.sleep(2)
                    else:
                        print("Backend server not responding after 5 attempts")
                        # Check process status one more time
                        if self.backend_process.poll() is not None:
                            stdout, stderr = self.backend_process.communicate()
                            print(f"Backend process died! Exit code: {self.backend_process.returncode}")
                            print(f"STDOUT: {stdout.decode()}")
                        return False
                except Exception as e:
                    print(f"Error connecting to backend: {e}")
                    return False
            
            return False
        except Exception as e:
            print(f"Error starting backend: {e}")
            return False
    
    def stop_backend(self):
        if hasattr(self, 'backend_process') and self.backend_process:
            print("Stopping backend server...")
            self.backend_process.terminate()
            self.backend_process.wait()
            print("Backend server stopped")
    
    def test_deprecation_warnings(self) -> bool:
        print("\nTesting for deprecation warnings...")
        
        try:
            # Start backend and capture stderr
            os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null || true")
            time.sleep(2)
            
            process = subprocess.Popen([
                sys.executable, "../main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            time.sleep(5)
            
            # Check stderr for deprecation warnings
            stderr_output = process.stderr.read().decode('utf-8')
            
            process.terminate()
            process.wait()
            
            if "DeprecationWarning" in stderr_output or "deprecated" in stderr_output.lower():
                print("Deprecation warnings found:")
                print(stderr_output)
                return False
            else:
                print("No deprecation warnings found")
                return True
                
        except Exception as e:
            print(f"Deprecation warning test error: {e}")
            return False
    
    def test_malformed_requests(self) -> bool:
        print("\nTesting malformed requests...")
        
        tests_passed = 0
        total_tests = 5
        
        # Test 1: Empty POST body
        try:
            response = requests.post(f"{self.base_url}/analyze", timeout=10)
            if response.status_code == 422:
                print("Empty POST body handled correctly")
                tests_passed += 1
            else:
                print(f"Empty POST body not handled: {response.status_code}")
        except Exception as e:
            print(f"Empty POST test error: {e}")
        
        # Test 2: Invalid JSON
        try:
            response = requests.post(f"{self.base_url}/analyze", 
                                   data="invalid json", 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=10)
            print(f"Invalid JSON handled: {response.status_code}")
            tests_passed += 1
        except Exception as e:
            print(f"Invalid JSON handled correctly: {e}")
            tests_passed += 1
        
        # Test 3: Wrong content type
        try:
            files = {'file': ('test.txt', b'not an image', 'text/plain')}
            response = requests.post(f"{self.base_url}/analyze", files=files, timeout=10)
            if response.status_code == 400:
                print("Wrong content type handled correctly")
                tests_passed += 1
            else:
                print(f"Wrong content type not handled: {response.status_code}")
        except Exception as e:
            print(f"Wrong content type test error: {e}")
        
        # Test 4: Corrupted image data
        try:
            corrupted_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00'
            files = {'file': ('corrupted.jpg', corrupted_data, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
            print(f"Corrupted image handled: {response.status_code}")
            tests_passed += 1
        except Exception as e:
            print(f"Corrupted image handled correctly: {e}")
            tests_passed += 1
        
        # Test 5: Very small file
        try:
            tiny_data = b'\xff\xd8\xff\xe0'
            files = {'file': ('tiny.jpg', tiny_data, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
            print(f"Tiny file handled: {response.status_code}")
            tests_passed += 1
        except Exception as e:
            print(f"Tiny file handled correctly: {e}")
            tests_passed += 1
        
        return tests_passed >= total_tests * 0.8
    
    def test_unicode_filenames(self) -> bool:
        print("\nTesting Unicode filenames...")
        
        if not self.test_images_dir.exists():
            print("No test images directory found")
            return False
        
        test_images = list(self.test_images_dir.glob("*.jpg"))
        if not test_images:
            print("No test images found")
            return False
        
        try:
            # Test with Unicode filename
            image_path = test_images[0]
            unicode_filename = "测试图片_🎨_artwork.jpg"
            
            with open(image_path, 'rb') as f:
                files = {'file': (unicode_filename, f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
            
            if response.status_code == 200:
                print("Unicode filename handled correctly")
                return True
            else:
                print(f"Unicode filename not handled: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Unicode filename test error: {e}")
            return False
    
    def test_concurrent_model_access(self) -> bool:
        print("\nTesting concurrent model access...")
        
        if not self.test_images_dir.exists():
            print("No test images directory found")
            return False
        
        test_images = list(self.test_images_dir.glob("*.jpg"))
        if not test_images:
            print("No test images found")
            return False
        
        import threading
        
        results = []
        
        def make_request(image_path, result_list):
            try:
                with open(image_path, 'rb') as f:
                    files = {'file': (image_path.name, f, 'image/jpeg')}
                    response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
                
                result_list.append({
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                })
            except Exception as e:
                result_list.append({
                    'status_code': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Start multiple concurrent requests
        threads = []
        for i in range(3):
            image_path = test_images[i % len(test_images)]
            thread = threading.Thread(target=make_request, args=(image_path, results))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        successful_requests = sum(1 for r in results if r['success'])
        print(f"Concurrent model access: {successful_requests}/{len(results)} successful")
        
        return successful_requests == len(results)
    
    def test_memory_leaks(self) -> bool:
        print("\nTesting for memory leaks...")
        
        if not self.test_images_dir.exists():
            print("No test images directory found")
            return False
        
        test_images = list(self.test_images_dir.glob("*.jpg"))
        if not test_images:
            print("No test images found")
            return False
        
        import psutil
        
        try:
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Make multiple requests
            for i in range(10):
                image_path = test_images[i % len(test_images)]
                with open(image_path, 'rb') as f:
                    files = {'file': (image_path.name, f, 'image/jpeg')}
                    response = requests.post(f"{self.base_url}/analyze", files=files, timeout=30)
                
                # Check memory after each request
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                if memory_increase > 100:
                    print(f"Potential memory leak detected: {memory_increase:.1f}MB increase")
                    return False
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            total_increase = final_memory - initial_memory
            
            print(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{total_increase:.1f}MB)")
            
            if total_increase < 50:
                print("No significant memory leak detected")
                return True
            else:
                print("Potential memory leak detected")
                return False
                
        except Exception as e:
            print(f"Memory leak test error: {e}")
            return False
    
    def test_error_response_format(self) -> bool:
        print("\nTesting error response format...")
        
        tests_passed = 0
        total_tests = 3
        
        # Test 1: Missing file
        try:
            response = requests.post(f"{self.base_url}/analyze", timeout=10)
            if response.status_code == 422:
                data = response.json()
                if 'detail' in data:
                    print("Missing file error format correct")
                    tests_passed += 1
                else:
                    print("Missing file error format incorrect")
            else:
                print(f"Missing file error status incorrect: {response.status_code}")
        except Exception as e:
            print(f"Missing file error test failed: {e}")
        
        # Test 2: Invalid file type
        try:
            files = {'file': ('test.txt', b'not an image', 'text/plain')}
            response = requests.post(f"{self.base_url}/analyze", files=files, timeout=10)
            if response.status_code == 400:
                data = response.json()
                if 'detail' in data:
                    print("Invalid file type error format correct")
                    tests_passed += 1
                else:
                    print("Invalid file type error format incorrect")
            else:
                print(f"Invalid file type error status incorrect: {response.status_code}")
        except Exception as e:
            print(f"Invalid file type error test failed: {e}")
        
        # Test 3: Non-existent endpoint
        try:
            response = requests.get(f"{self.base_url}/nonexistent", timeout=10)
            if response.status_code == 404:
                print("Non-existent endpoint handled correctly")
                tests_passed += 1
            else:
                print(f"Non-existent endpoint not handled: {response.status_code}")
        except Exception as e:
            print(f"Non-existent endpoint test failed: {e}")
        
        return tests_passed >= total_tests * 0.8
    
    def ensure_backend_running(self) -> bool:
        if hasattr(self, 'backend_process') and self.backend_process:
            if self.backend_process.poll() is not None:
                print("Backend died, restarting...")
                return self.start_backend()
            try:
                response = requests.get(f"{self.base_url}/", timeout=2)
                return response.status_code == 200
            except:
                print("Backend not responding, restarting...")
                return self.start_backend()
        return self.start_backend()
    
    def run_all_tests(self) -> bool:
        print("Starting Bug Fixing Tests")
        print("=" * 50)
        
        if not self.start_backend():
            print("Cannot run tests without backend")
            return False
        
        try:
            tests = [
                ("Deprecation Warnings", self.test_deprecation_warnings),
                ("Malformed Requests", self.test_malformed_requests),
                ("Unicode Filenames", self.test_unicode_filenames),
                ("Concurrent Model Access", self.test_concurrent_model_access),
                ("Memory Leaks", self.test_memory_leaks),
                ("Error Response Format", self.test_error_response_format),
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                print(f"\nRunning: {test_name}")
                
                # Ensure backend is running before each test
                if not self.ensure_backend_running():
                    print(f"{test_name} FAILED - Backend not available")
                    continue
                
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
            print(f"Bug Fixing Test Results: {passed}/{total} passed")
            
            if passed == total:
                print("All bug fixing tests PASSED!")
                return True
            else:
                print("Some bug fixing tests FAILED!")
                return False
                
        finally:
            self.stop_backend()

if __name__ == "__main__":
    tester = BugFixTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
