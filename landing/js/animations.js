/**
 * HybridRAG Landing Page - Scroll Animations
 * Uses IntersectionObserver for performant scroll-triggered animations
 */

(function() {
    'use strict';

    // Configuration
    const config = {
        revealThreshold: 0.1,
        revealRootMargin: '0px 0px -50px 0px',
        staggerDelay: 100
    };

    /**
     * Initialize reveal animations using IntersectionObserver
     */
    function initRevealAnimations() {
        const revealElements = document.querySelectorAll('.reveal');

        if (!revealElements.length) return;

        // Check for IntersectionObserver support
        if (!('IntersectionObserver' in window)) {
            // Fallback: show all elements immediately
            revealElements.forEach(el => el.classList.add('revealed'));
            return;
        }

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    // Add revealed class to trigger animation
                    entry.target.classList.add('revealed');

                    // Unobserve after revealing (animation only happens once)
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: config.revealThreshold,
            rootMargin: config.revealRootMargin
        });

        // Observe all reveal elements
        revealElements.forEach(el => observer.observe(el));
    }

    /**
     * Initialize staggered animations for grid children
     */
    function initStaggeredAnimations() {
        const grids = document.querySelectorAll('.features__grid, .modes__grid');

        grids.forEach(grid => {
            const children = grid.children;

            // Assign stagger delays
            Array.from(children).forEach((child, index) => {
                if (!child.dataset.delay) {
                    child.style.transitionDelay = `${index * config.staggerDelay}ms`;
                }
            });
        });
    }

    /**
     * Initialize parallax effects
     */
    function initParallax() {
        const parallaxElements = document.querySelectorAll('.hero__gradient');

        if (!parallaxElements.length || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            return;
        }

        let ticking = false;

        function updateParallax() {
            const scrollY = window.scrollY;

            parallaxElements.forEach((el, index) => {
                const speed = (index + 1) * 0.1;
                const yPos = scrollY * speed;
                el.style.transform = `translateY(${yPos}px)`;
            });

            ticking = false;
        }

        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(updateParallax);
                ticking = true;
            }
        }, { passive: true });
    }

    /**
     * Initialize counter animations
     */
    function initCounters() {
        const counters = document.querySelectorAll('.hero__stat-value');

        if (!counters.length) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        counters.forEach(counter => observer.observe(counter));
    }

    /**
     * Animate a counter element
     */
    function animateCounter(element) {
        const text = element.textContent;
        const match = text.match(/(\d+)/);

        if (!match) return;

        const target = parseInt(match[1], 10);
        const suffix = text.replace(match[0], '');
        const duration = 1500;
        const start = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function (easeOutExpo)
            const easeProgress = 1 - Math.pow(2, -10 * progress);
            const current = Math.floor(easeProgress * target);

            element.textContent = current + suffix;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = target + suffix;
            }
        }

        requestAnimationFrame(update);
    }

    /**
     * Initialize text typing effect
     */
    function initTypingEffect() {
        const typingElements = document.querySelectorAll('[data-typing]');

        typingElements.forEach(element => {
            const text = element.dataset.typing;
            element.textContent = '';

            let index = 0;
            const interval = setInterval(() => {
                if (index < text.length) {
                    element.textContent += text[index];
                    index++;
                } else {
                    clearInterval(interval);
                }
            }, 50);
        });
    }

    /**
     * Initialize magnetic effect on buttons
     */
    function initMagneticEffect() {
        const magneticElements = document.querySelectorAll('.btn--primary');

        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            return;
        }

        magneticElements.forEach(element => {
            element.addEventListener('mousemove', (e) => {
                const rect = element.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;

                element.style.transform = `translate(${x * 0.1}px, ${y * 0.1}px)`;
            });

            element.addEventListener('mouseleave', () => {
                element.style.transform = '';
            });
        });
    }

    /**
     * Initialize graph node animations
     */
    function initGraphAnimations() {
        const graph = document.querySelector('.hero__graph-svg');

        if (!graph) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    graph.classList.add('animated');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.3 });

        observer.observe(graph);
    }

    /**
     * Initialize section progress indicator
     */
    function initSectionProgress() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav__link[href^="#"]');

        if (!sections.length || !navLinks.length) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');

                    navLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, {
            threshold: 0.3,
            rootMargin: '-100px 0px -50% 0px'
        });

        sections.forEach(section => observer.observe(section));
    }

    /**
     * Initialize scroll indicator
     */
    function initScrollIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'scroll-progress';
        indicator.innerHTML = '<div class="scroll-progress__bar"></div>';
        document.body.appendChild(indicator);

        const bar = indicator.querySelector('.scroll-progress__bar');

        // Add styles dynamically
        const style = document.createElement('style');
        style.textContent = `
            .scroll-progress {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                z-index: 9999;
                background: rgba(255, 255, 255, 0.1);
            }
            .scroll-progress__bar {
                height: 100%;
                background: linear-gradient(90deg, #8B5CF6, #3B82F6);
                width: 0%;
                transition: width 0.1s linear;
            }
        `;
        document.head.appendChild(style);

        function updateProgress() {
            const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPosition = window.scrollY;
            const progress = (scrollPosition / scrollHeight) * 100;
            bar.style.width = `${progress}%`;
        }

        window.addEventListener('scroll', updateProgress, { passive: true });
        updateProgress();
    }

    /**
     * Initialize all animations
     */
    function init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initAll);
        } else {
            initAll();
        }
    }

    function initAll() {
        initRevealAnimations();
        initStaggeredAnimations();
        initParallax();
        initCounters();
        initMagneticEffect();
        initGraphAnimations();
        initSectionProgress();
        // Optionally enable scroll indicator
        // initScrollIndicator();
    }

    // Initialize
    init();

    // Export for external use
    window.HybridRAGAnimations = {
        initRevealAnimations,
        initStaggeredAnimations,
        initParallax,
        initCounters,
        reinit: initAll
    };

})();
