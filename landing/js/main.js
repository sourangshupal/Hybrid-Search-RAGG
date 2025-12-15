/**
 * HybridRAG Landing Page - Main JavaScript
 * Handles interactions, smooth scroll, and UI functionality
 */

(function() {
    'use strict';

    // ============================================
    // Smooth Scroll with Lenis
    // ============================================

    let lenis;

    function initSmoothScroll() {
        // Check if Lenis is available
        if (typeof Lenis === 'undefined') {
            console.warn('Lenis not loaded, using native smooth scroll');
            return;
        }

        // Initialize Lenis
        lenis = new Lenis({
            duration: 1.2,
            easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
            orientation: 'vertical',
            gestureOrientation: 'vertical',
            smoothWheel: true,
            wheelMultiplier: 1,
            touchMultiplier: 2,
            infinite: false,
        });

        // RAF loop
        function raf(time) {
            lenis.raf(time);
            requestAnimationFrame(raf);
        }
        requestAnimationFrame(raf);

        // Handle anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href === '#') return;

                e.preventDefault();
                const target = document.querySelector(href);

                if (target) {
                    lenis.scrollTo(target, {
                        offset: -80, // Account for fixed nav
                        duration: 1.5
                    });

                    // Close mobile menu if open
                    closeMobileMenu();
                }
            });
        });
    }

    // ============================================
    // Navigation
    // ============================================

    function initNavigation() {
        const nav = document.querySelector('.nav');
        const menuBtn = document.getElementById('menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');

        if (!nav) return;

        // Scroll behavior
        let lastScrollY = window.scrollY;
        let ticking = false;

        function updateNav() {
            const scrollY = window.scrollY;

            // Add scrolled class
            if (scrollY > 50) {
                nav.classList.add('nav--scrolled');
            } else {
                nav.classList.remove('nav--scrolled');
            }

            // Hide/show on scroll (optional)
            // if (scrollY > lastScrollY && scrollY > 200) {
            //     nav.style.transform = 'translateY(-100%)';
            // } else {
            //     nav.style.transform = 'translateY(0)';
            // }

            lastScrollY = scrollY;
            ticking = false;
        }

        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(updateNav);
                ticking = true;
            }
        }, { passive: true });

        // Mobile menu toggle
        if (menuBtn && mobileMenu) {
            menuBtn.addEventListener('click', () => {
                menuBtn.classList.toggle('active');
                mobileMenu.classList.toggle('active');
                document.body.style.overflow = mobileMenu.classList.contains('active') ? 'hidden' : '';
            });

            // Close on link click
            mobileMenu.querySelectorAll('a').forEach(link => {
                link.addEventListener('click', closeMobileMenu);
            });

            // Close on escape
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && mobileMenu.classList.contains('active')) {
                    closeMobileMenu();
                }
            });
        }
    }

    function closeMobileMenu() {
        const menuBtn = document.getElementById('menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');

        if (menuBtn && mobileMenu) {
            menuBtn.classList.remove('active');
            mobileMenu.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    // ============================================
    // Copy to Clipboard
    // ============================================

    function initCopyButtons() {
        // Code block copy button
        const copyBtn = document.getElementById('copy-btn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const codeBlock = document.querySelector('.code-block__code code');
                if (codeBlock) {
                    copyToClipboard(codeBlock.textContent, copyBtn);
                }
            });
        }

        // Install command copy button
        const installCopyBtn = document.getElementById('install-copy');
        if (installCopyBtn) {
            installCopyBtn.addEventListener('click', () => {
                const command = document.querySelector('.cta-card__command');
                if (command) {
                    copyToClipboard(command.textContent, installCopyBtn);
                }
            });
        }
    }

    async function copyToClipboard(text, button) {
        try {
            await navigator.clipboard.writeText(text);
            showCopiedFeedback(button);
        } catch (err) {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();

            try {
                document.execCommand('copy');
                showCopiedFeedback(button);
            } catch (e) {
                console.error('Failed to copy:', e);
            }

            document.body.removeChild(textarea);
        }
    }

    function showCopiedFeedback(button) {
        const originalHTML = button.innerHTML;
        const spanElement = button.querySelector('span');

        button.classList.add('copied');

        if (spanElement) {
            spanElement.textContent = 'Copied!';
        }

        setTimeout(() => {
            button.classList.remove('copied');
            if (spanElement) {
                spanElement.textContent = 'Copy';
            }
        }, 2000);
    }

    // ============================================
    // Theme Toggle (if needed in future)
    // ============================================

    function initThemeToggle() {
        const themeToggle = document.getElementById('theme-toggle');
        if (!themeToggle) return;

        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);

        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }

    // ============================================
    // Keyboard Navigation
    // ============================================

    function initKeyboardNav() {
        // Skip to main content
        const skipLink = document.createElement('a');
        skipLink.href = '#features';
        skipLink.className = 'skip-link';
        skipLink.textContent = 'Skip to main content';
        skipLink.style.cssText = `
            position: fixed;
            top: -100px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--primary-violet);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            z-index: 9999;
            transition: top 0.3s;
        `;

        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '20px';
        });

        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-100px';
        });

        document.body.insertBefore(skipLink, document.body.firstChild);

        // Focus visible polyfill behavior
        document.body.addEventListener('mousedown', () => {
            document.body.classList.add('using-mouse');
        });

        document.body.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.remove('using-mouse');
            }
        });
    }

    // ============================================
    // Performance Optimizations
    // ============================================

    function initLazyLoading() {
        // Native lazy loading is used in HTML
        // This adds fallback for older browsers if needed
        if ('loading' in HTMLImageElement.prototype) {
            // Native lazy loading supported
            return;
        }

        // Fallback using IntersectionObserver
        const images = document.querySelectorAll('img[loading="lazy"]');

        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    imageObserver.unobserve(img);
                }
            });
        });

        images.forEach(img => imageObserver.observe(img));
    }

    // ============================================
    // Analytics Events (placeholder)
    // ============================================

    function initAnalytics() {
        // Track CTA clicks
        document.querySelectorAll('.btn--primary').forEach(btn => {
            btn.addEventListener('click', () => {
                // Send analytics event
                if (typeof gtag !== 'undefined') {
                    gtag('event', 'click', {
                        event_category: 'CTA',
                        event_label: btn.textContent.trim()
                    });
                }
            });
        });

        // Track section views
        const sections = document.querySelectorAll('section[id]');

        if ('IntersectionObserver' in window) {
            const sectionObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const sectionId = entry.target.getAttribute('id');
                        // Send analytics event
                        if (typeof gtag !== 'undefined') {
                            gtag('event', 'view_section', {
                                section_name: sectionId
                            });
                        }
                    }
                });
            }, { threshold: 0.5 });

            sections.forEach(section => sectionObserver.observe(section));
        }
    }

    // ============================================
    // Error Handling
    // ============================================

    function handleErrors() {
        window.addEventListener('error', (e) => {
            console.error('Error:', e.message);
            // Could send to error tracking service
        });

        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e.reason);
        });
    }

    // ============================================
    // Initialize Everything
    // ============================================

    function init() {
        // Wait for DOM
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initAll);
        } else {
            initAll();
        }
    }

    function initAll() {
        handleErrors();
        initSmoothScroll();
        initNavigation();
        initCopyButtons();
        initKeyboardNav();
        initLazyLoading();
        // initAnalytics(); // Enable when analytics is set up

        console.log('HybridRAG Landing Page initialized');
    }

    // Start
    init();

    // Export for external use
    window.HybridRAG = {
        lenis,
        closeMobileMenu,
        copyToClipboard,
        reinit: initAll
    };

})();
