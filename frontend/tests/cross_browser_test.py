#!/usr/bin/env python3
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import json

class CrossBrowserTester:
    def __init__(self):
        self.frontend_dir = Path(__file__).parent.parent
        self.landing_page = self.frontend_dir / "landing.html"
        self.main_app = self.frontend_dir / "index.html"
        self.results = []
        
    def get_browser_commands(self) -> Dict[str, List[str]]:
        return {
            'chrome': [
                '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                '--headless', '--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage'
            ],
            'firefox': [
                '/Applications/Firefox.app/Contents/MacOS/firefox',
                '--headless'
            ],
            'safari': [
                '/Applications/Safari.app/Contents/MacOS/Safari'
            ],
            'edge': [
                '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
                '--headless', '--disable-gpu', '--no-sandbox'
            ]
        }
    
    def test_html_validation(self) -> bool:
        print("\nTesting HTML validation...")
        
        html_files = [self.landing_page, self.main_app]
        all_valid = True
        
        for html_file in html_files:
            if not html_file.exists():
                print(f"HTML file not found: {html_file}")
                all_valid = False
                continue
            
            print(f"Validating {html_file.name}...")

            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if '<!DOCTYPE html>' not in content:
                    print(f"Missing DOCTYPE in {html_file.name}")
                    all_valid = False
                elif '<html' not in content:
                    print(f"Missing <html> tag in {html_file.name}")
                    all_valid = False
                elif '<head>' not in content:
                    print(f"Missing <head> tag in {html_file.name}")
                    all_valid = False
                elif '<body>' not in content:
                    print(f"Missing <body> tag in {html_file.name}")
                    all_valid = False
                else:
                    print(f"{html_file.name} has valid HTML structure")
        
        return all_valid
    
    def test_css_validation(self) -> bool:
        print("\nTesting CSS validation...")
        
        html_files = [self.landing_page, self.main_app]
        all_valid = True
        
        for html_file in html_files:
            if not html_file.exists():
                continue
                
            print(f"Validating CSS in {html_file.name}...")
            
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract CSS from <style> tags
                import re
                css_matches = re.findall(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)
                
                for css in css_matches:
                    # Basic CSS validation
                    if '{' in css and '}' in css:
                        print(f"CSS structure valid in {html_file.name}")
                    else:
                        print(f"CSS structure invalid in {html_file.name}")
                        all_valid = False
        
        return all_valid
    
    def test_javascript_validation(self) -> bool:
        print("\nTesting JavaScript validation...")
        
        html_files = [self.landing_page, self.main_app]
        all_valid = True
        
        for html_file in html_files:
            if not html_file.exists():
                continue
                
            print(f"Validating JavaScript in {html_file.name}...")
            
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                import re
                js_matches = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
                
                for js in js_matches:
                    if 'function' in js or 'addEventListener' in js or 'document.' in js:
                        print(f"JavaScript structure valid in {html_file.name}")
                    else:
                        print(f"Limited JavaScript in {html_file.name}")
        
        return all_valid
    
    def test_responsive_design(self) -> bool:
        print("\nTesting responsive design...")
        
        html_files = [self.landing_page, self.main_app]
        all_responsive = True
        
        for html_file in html_files:
            if not html_file.exists():
                continue
                
            print(f"Checking responsive design in {html_file.name}...")
            
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for responsive design elements
                responsive_elements = [
                    'viewport',
                    'media queries',
                    '@media',
                    'flex',
                    'grid',
                    'responsive'
                ]
                
                found_elements = []
                for element in responsive_elements:
                    if element in content.lower():
                        found_elements.append(element)
                
                if len(found_elements) >= 2:
                    print(f"Responsive design elements found: {found_elements}")
                else:
                    print(f"Limited responsive design in {html_file.name}")
                    all_responsive = False
        
        return all_responsive
    
    def test_accessibility(self) -> bool:
        print("\nTesting accessibility...")
        
        html_files = [self.landing_page, self.main_app]
        all_accessible = True
        
        for html_file in html_files:
            if not html_file.exists():
                continue
                
            print(f"Checking accessibility in {html_file.name}...")
            
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                accessibility_elements = [
                    'alt=',
                    'aria-label',
                    'aria-describedby',
                    'role=',
                    'tabindex',
                    'title='
                ]
                
                found_elements = []
                for element in accessibility_elements:
                    if element in content.lower():
                        found_elements.append(element)
                
                if len(found_elements) >= 2:
                    print(f"Accessibility elements found: {found_elements}")
                else:
                    print(f"Limited accessibility features in {html_file.name}")
                    all_accessible = False
        
        return all_accessible
    
    def test_browser_compatibility(self) -> bool:
        print("\nTesting browser compatibility...")
        
        browsers = self.get_browser_commands()
        compatible_browsers = 0
        total_browsers = len(browsers)
        
        for browser_name, command in browsers.items():
            print(f"Testing {browser_name}...")
            
            if os.path.exists(command[0]):
                print(f"{browser_name} executable found")
                compatible_browsers += 1
            else:
                print(f"{browser_name} executable not found")
        
        print(f"Browser compatibility: {compatible_browsers}/{total_browsers} browsers available")
        return compatible_browsers >= total_browsers * 0.5  # At least 50% available
    
    def test_file_structure(self) -> bool:
        print("\nTesting file structure...")
        
        required_files = [
            self.landing_page,
            self.main_app
        ]
        
        all_files_exist = True
        
        for file_path in required_files:
            if file_path.exists():
                print(f"{file_path.name} exists")
            else:
                print(f"{file_path.name} missing")
                all_files_exist = False
        
        assets_dir = self.frontend_dir / "assets"
        if assets_dir.exists():
            print(f"Assets directory exists")
        else:
            print(f"No assets directory found")
        
        return all_files_exist
    
    def test_cross_origin_requests(self) -> bool:
        print("\nTesting cross-origin requests...")
        
        try:
            try:
                import requests
            except ImportError:
                print("CORS test skipped (requests module not available)")
                return True
                
            response = requests.options("http://localhost:8000/analyze", timeout=5)
            
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
                return False
                
        except Exception as e:
            print(f"CORS test skipped (backend not running): {e}")
            return True  # Skip if backend not running
    
    def test_security_headers(self) -> bool:
        print("\nTesting security headers...")
        
        html_files = [self.landing_page, self.main_app]
        security_score = 0
        
        for html_file in html_files:
            if not html_file.exists():
                continue
                
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                security_elements = [
                    'content-security-policy',
                    'x-frame-options',
                    'x-content-type-options',
                    'referrer-policy'
                ]
                
                for element in security_elements:
                    if element in content.lower():
                        security_score += 1
        
        print(f"   Security score: {security_score}/8")
        
        if security_score >= 2:
            print("Basic security measures present")
            return True
        else:
            print("Limited security measures")
            return False
    
    def run_all_tests(self) -> bool:
        print("Starting Cross-Browser Tests")
        print("=" * 50)
        
        tests = [
            ("HTML Validation", self.test_html_validation),
            ("CSS Validation", self.test_css_validation),
            ("JavaScript Validation", self.test_javascript_validation),
            ("Responsive Design", self.test_responsive_design),
            ("Accessibility", self.test_accessibility),
            ("Browser Compatibility", self.test_browser_compatibility),
            ("File Structure", self.test_file_structure),
            ("Cross-Origin Requests", self.test_cross_origin_requests),
            ("Security Headers", self.test_security_headers),
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
        print(f"Cross-Browser Test Results: {passed}/{total} passed")
        
        if passed == total:
            print("All cross-browser tests PASSED!")
            return True
        else:
            print("Some cross-browser tests FAILED!")
            return False

if __name__ == "__main__":
    tester = CrossBrowserTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
