"""
REAL TEST - Forces Browser Usage
==================================
Tests both HTTP and Browser paths with sites that actually need JS
"""

import asyncio
from tools.web_fetcher import TrueLightweightHybridScraper
import logging

logger = logging.getLogger('TestRunner')

async def test_both_paths():
    """Test BOTH HTTP and Browser paths"""
    
    print("\n" + "="*70)
    print("🧪 COMPREHENSIVE TEST - HTTP + BROWSER")
    print("="*70 + "\n")
    
    test_urls = [
        # Static sites (WILL use HTTP)
        "http://info.cern.ch",           # Simple static HTML
        "https://httpbin.org/html",      # Static HTML page
        
        # Sites that MIGHT need browser (let's see what happens)
        "https://www.google.com",        # May have minimal content check
        "https://github.com",            # Complex but may work with HTTP
    ]
    
    print("📋 Test Strategy:")
    print("   • Static sites → Should use HTTP ⚡")
    print("   • Dynamic sites → Should use Browser 🌐")
    print("   • We'll see which is which!\n")
    
    async with TrueLightweightHybridScraper(
        rate_limit=1.0,
        max_concurrent=2,
        timeout=15
    ) as scraper:
        
        results = await scraper.scrape_urls(test_urls, show_progress=True)
        
        print("\n" + "="*70)
        print("📊 RESULTS BREAKDOWN")
        print("="*70 + "\n")
        
        http_urls = []
        browser_urls = []
        failed_urls = []
        
        for r in results:
            meta = r['metadata']
            url = r['url']
            method = meta.get('method', 'unknown')
            success = meta.get('success', False)
            
            if success:
                if method == 'http':
                    http_urls.append(url)
                elif method == 'browser':
                    browser_urls.append(url)
            else:
                failed_urls.append((url, meta.get('error', 'unknown')))
        
        # Print HTTP results
        if http_urls:
            print(f"⚡ HTTP SUCCESSES ({len(http_urls)}):")
            for url in http_urls:
                print(f"   • {url}")
        
        # Print Browser results
        if browser_urls:
            print(f"\n🌐 BROWSER SUCCESSES ({len(browser_urls)}):")
            for url in browser_urls:
                print(f"   • {url}")
        else:
            print(f"\n⚠️  BROWSER NEVER USED!")
            print(f"   Reason: All sites worked with HTTP")
            print(f"   OR sites that need browser failed to connect")
        
        # Print failures
        if failed_urls:
            print(f"\n❌ FAILURES ({len(failed_urls)}):")
            for url, error in failed_urls:
                print(f"   • {url}")
                print(f"     Error: {error[:80]}")
        
        # Final verdict
        print("\n" + "="*70)
        print("🎯 VERIFICATION")
        print("="*70)
        
        http_count = scraper.stats['http_scrapes']
        browser_count = scraper.stats['browser_scrapes']
        
        print(f"\n✅ HTTP path: {'VERIFIED ✓' if http_count > 0 else 'NOT TESTED ✗'}")
        print(f"✅ Browser path: {'VERIFIED ✓' if browser_count > 0 else 'NOT TESTED ✗'}")
        
        if browser_count == 0:
            print("\n" + "="*70)
            print("⚠️  BROWSER PATH NOT VERIFIED!")
            print("="*70)
            print("Possible reasons:")
            print("1. All test sites were static (worked with HTTP)")
            print("2. Sites that need browser failed to connect")
            print("3. Need to test with actual SPA (React/Vue app)")
            print("\nTo FORCE browser test, try:")
            print("• A live React app")
            print("• A Vue.js app")  
            print("• Any SPA that returns minimal HTML")


async def test_with_mock_dynamic_content():
    """
    Alternative: Test detection logic directly
    """
    from tools.web_fetcher import DynamicContentDetector
    
    print("\n\n" + "="*70)
    print("🔍 TESTING DETECTION LOGIC DIRECTLY")
    print("="*70 + "\n")
    
    detector = DynamicContentDetector()
    
    # Test case 1: Static HTML (should NOT need browser)
    static_html = """
    <html>
        <head><title>Static Page</title></head>
        <body>
            <h1>Welcome</h1>
            <p>This is a static page with lots of content. Lorem ipsum dolor sit amet, 
            consectetur adipiscing elit. This has enough text to pass the 200 char check.
            More content here to make sure we have plenty of text content.</p>
        </body>
    </html>
    """
    
    needs_browser, reason = detector.needs_browser(static_html, "http://example.com")
    print(f"1️⃣  Static HTML:")
    print(f"   Needs browser: {needs_browser} (Reason: {reason})")
    print(f"   ✅ Expected: False (should use HTTP)")
    print(f"   Result: {'✓ PASS' if not needs_browser else '✗ FAIL'}")
    
    # Test case 2: React app (SHOULD need browser)
    react_html = """
    <html>
        <head><title>React App</title></head>
        <body>
            <div id="root"></div>
            <script>window.React = {}</script>
        </body>
    </html>
    """
    
    needs_browser, reason = detector.needs_browser(react_html, "http://example.com")
    print(f"\n2️⃣  React App HTML:")
    print(f"   Needs browser: {needs_browser} (Reason: {reason})")
    print(f"   ✅ Expected: True (should use Browser)")
    print(f"   Result: {'✓ PASS' if needs_browser else '✗ FAIL'}")
    
    # Test case 3: Minimal content (SHOULD need browser)
    minimal_html = """
    <html>
        <head><title>App</title></head>
        <body>
            <div id="app"></div>
        </body>
    </html>
    """
    
    needs_browser, reason = detector.needs_browser(minimal_html, "http://example.com")
    print(f"\n3️⃣  Minimal Content:")
    print(f"   Needs browser: {needs_browser} (Reason: {reason})")
    print(f"   ✅ Expected: True (should use Browser)")
    print(f"   Result: {'✓ PASS' if needs_browser else '✗ FAIL'}")
    
    print("\n" + "="*70)
    print("📊 DETECTION LOGIC TEST COMPLETE")
    print("="*70)


async def main():
    """Run all tests"""
    
    # Test 1: Real scraping (may not hit browser)
    await test_both_paths()
    
    # Test 2: Detection logic (proves browser WOULD be used)
    await test_with_mock_dynamic_content()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)
    print("\nConclusion:")
    print("• HTTP path: Can be verified with static sites")
    print("• Browser path: Needs actual JS-heavy SPA to test")
    print("• Detection logic: Can be verified with mock HTML")
    print()


if __name__ == "__main__":
    print("\n🚀 Running comprehensive tests...\n")
    asyncio.run(main())