// Advanced Animation Controller for ExoPlanet AI

class AnimationController {
    constructor() {
        this.animationQueue = [];
        this.isAnimating = false;
        this.observers = new Map();
        
        this.init();
    }

    init() {
        this.setupIntersectionObserver();
        this.setupScrollAnimations();
        this.setupHoverEffects();
        this.setupParallaxEffects();
        this.setupTypingAnimations();
    }

    // Intersection Observer for scroll-triggered animations
    setupIntersectionObserver() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        this.scrollObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.triggerElementAnimation(entry.target);
                }
            });
        }, observerOptions);

        // Observe elements that should animate on scroll
        const animatedElements = document.querySelectorAll(
            '.analytics-card, .telescope-card, .stat-card, .tech-item, .feature-item, .training-item'
        );
        
        animatedElements.forEach((el, index) => {
            el.classList.add('scroll-reveal');
            el.style.animationDelay = `${index * 0.1}s`;
            this.scrollObserver.observe(el);
        });
    }

    triggerElementAnimation(element) {
        const animationType = element.dataset.animation || 'slideInUp';
        element.classList.add(`animate-${animationType}`);
        
        // Add stagger effect for grouped elements
        if (element.parentElement.classList.contains('analytics-grid')) {
            const siblings = Array.from(element.parentElement.children);
            const index = siblings.indexOf(element);
            element.style.animationDelay = `${index * 0.2}s`;
        }
    }

    // Scroll-based animations
    setupScrollAnimations() {
        let ticking = false;

        const updateScrollAnimations = () => {
            this.updateParallaxElements();
            this.updateProgressBars();
            this.updateCounterAnimations();
            ticking = false;
        };

        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(updateScrollAnimations);
                ticking = true;
            }
        });
    }

    updateParallaxElements() {
        const scrollY = window.pageYOffset;
        const parallaxElements = document.querySelectorAll('[data-parallax]');

        parallaxElements.forEach(element => {
            const speed = parseFloat(element.dataset.parallax) || 0.5;
            const yPos = -(scrollY * speed);
            element.style.transform = `translateY(${yPos}px)`;
        });
    }

    updateProgressBars() {
        const progressBars = document.querySelectorAll('.metric-fill, .feature-fill');
        const scrollY = window.pageYOffset;
        const windowHeight = window.innerHeight;

        progressBars.forEach(bar => {
            const rect = bar.getBoundingClientRect();
            const isVisible = rect.top < windowHeight && rect.bottom > 0;

            if (isVisible && !bar.classList.contains('animated')) {
                bar.classList.add('animated');
                const targetWidth = bar.dataset.width || bar.style.width;
                this.animateProgressBar(bar, targetWidth);
            }
        });
    }

    animateProgressBar(bar, targetWidth) {
        const duration = 1500;
        const startTime = performance.now();
        const startWidth = 0;

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            const currentWidth = startWidth + (parseFloat(targetWidth) - startWidth) * easeOutCubic;
            
            bar.style.width = currentWidth + '%';
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    updateCounterAnimations() {
        const counters = document.querySelectorAll('.stat-number, .metric-value');
        const scrollY = window.pageYOffset;
        const windowHeight = window.innerHeight;

        counters.forEach(counter => {
            const rect = counter.getBoundingClientRect();
            const isVisible = rect.top < windowHeight && rect.bottom > 0;

            if (isVisible && !counter.classList.contains('counted')) {
                counter.classList.add('counted');
                this.animateCounter(counter);
            }
        });
    }

    animateCounter(element) {
        const text = element.textContent;
        const number = parseFloat(text.replace(/[^\d.]/g, ''));
        
        if (isNaN(number)) return;

        const duration = 2000;
        const startTime = performance.now();
        const suffix = text.replace(/[\d.]/g, '');

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentNumber = number * easeOutQuart;
            
            element.textContent = currentNumber.toFixed(2) + suffix;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    // Hover effects
    setupHoverEffects() {
        // Add hover effects to interactive elements
        const hoverElements = document.querySelectorAll(
            '.telescope-card, .btn, .analytics-card, .stat-card, .tech-item'
        );

        hoverElements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                this.addHoverEffect(element);
            });

            element.addEventListener('mouseleave', () => {
                this.removeHoverEffect(element);
            });
        });
    }

    addHoverEffect(element) {
        element.classList.add('hover-lift');
        
        // Add subtle glow effect
        if (element.classList.contains('telescope-card') || element.classList.contains('analytics-card')) {
            element.classList.add('hover-glow');
        }
    }

    removeHoverEffect(element) {
        element.classList.remove('hover-lift', 'hover-glow');
    }

    // Parallax effects
    setupParallaxEffects() {
        // Add parallax data attributes to elements
        const planets = document.querySelectorAll('.planet');
        planets.forEach((planet, index) => {
            planet.dataset.parallax = 0.3 + (index * 0.1);
        });

        const stars = document.querySelector('.stars');
        if (stars) {
            stars.dataset.parallax = 0.1;
        }
    }

    // Typing animations
    setupTypingAnimations() {
        const typingElements = document.querySelectorAll('[data-typing]');
        
        typingElements.forEach(element => {
            const text = element.textContent;
            const speed = parseInt(element.dataset.typingSpeed) || 100;
            
            element.textContent = '';
            element.classList.add('text-typing');
            
            this.typeText(element, text, speed);
        });
    }

    typeText(element, text, speed) {
        let index = 0;
        
        const typeChar = () => {
            if (index < text.length) {
                element.textContent += text.charAt(index);
                index++;
                setTimeout(typeChar, speed);
            } else {
                element.classList.remove('text-typing');
            }
        };

        // Start typing after a delay
        setTimeout(typeChar, 1000);
    }

    // Advanced animation sequences
    playAnimationSequence(sequence) {
        this.animationQueue.push(sequence);
        this.processAnimationQueue();
    }

    processAnimationQueue() {
        if (this.isAnimating || this.animationQueue.length === 0) return;
        
        this.isAnimating = true;
        const sequence = this.animationQueue.shift();
        
        this.executeSequence(sequence).then(() => {
            this.isAnimating = false;
            this.processAnimationQueue();
        });
    }

    async executeSequence(sequence) {
        for (const step of sequence) {
            await this.executeAnimationStep(step);
        }
    }

    executeAnimationStep(step) {
        return new Promise((resolve) => {
            const { element, animation, duration = 500, delay = 0 } = step;
            
            setTimeout(() => {
                if (element && animation) {
                    element.classList.add(`animate-${animation}`);
                    
                    setTimeout(() => {
                        element.classList.remove(`animate-${animation}`);
                        resolve();
                    }, duration);
                } else {
                    resolve();
                }
            }, delay);
        });
    }

    // Particle system animations
    createParticleBurst(x, y, color = '#00d4ff') {
        const container = document.createElement('div');
        container.style.position = 'fixed';
        container.style.left = x + 'px';
        container.style.top = y + 'px';
        container.style.pointerEvents = 'none';
        container.style.zIndex = '9999';
        
        document.body.appendChild(container);

        // Create multiple particles
        for (let i = 0; i < 10; i++) {
            const particle = document.createElement('div');
            particle.style.position = 'absolute';
            particle.style.width = '4px';
            particle.style.height = '4px';
            particle.style.background = color;
            particle.style.borderRadius = '50%';
            particle.style.opacity = '1';
            
            const angle = (i / 10) * Math.PI * 2;
            const velocity = 50 + Math.random() * 50;
            const vx = Math.cos(angle) * velocity;
            const vy = Math.sin(angle) * velocity;
            
            container.appendChild(particle);
            
            // Animate particle
            this.animateParticle(particle, vx, vy);
        }

        // Remove container after animation
        setTimeout(() => {
            document.body.removeChild(container);
        }, 1000);
    }

    animateParticle(particle, vx, vy) {
        const duration = 1000;
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = elapsed / duration;
            
            if (progress < 1) {
                const x = vx * progress;
                const y = vy * progress - (progress * progress * 200); // Gravity effect
                const opacity = 1 - progress;
                
                particle.style.transform = `translate(${x}px, ${y}px)`;
                particle.style.opacity = opacity;
                
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    // Loading animations
    showLoadingAnimation(element) {
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        spinner.style.width = '40px';
        spinner.style.height = '40px';
        spinner.style.border = '3px solid var(--border-color)';
        spinner.style.borderTop = '3px solid var(--primary-color)';
        spinner.style.borderRadius = '50%';
        spinner.style.animation = 'spin 1s linear infinite';
        spinner.style.margin = '0 auto';
        
        element.innerHTML = '';
        element.appendChild(spinner);
    }

    hideLoadingAnimation(element, content) {
        element.innerHTML = content;
    }

    // Utility methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Initialize animation controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.animationController = new AnimationController();
});

// Add click effects to buttons
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('btn') || e.target.closest('.btn')) {
        const button = e.target.classList.contains('btn') ? e.target : e.target.closest('.btn');
        const rect = button.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        if (window.animationController) {
            window.animationController.createParticleBurst(x, y);
        }
    }
});

// Add special effects for telescope card selection
document.addEventListener('click', (e) => {
    if (e.target.closest('.telescope-card')) {
        const card = e.target.closest('.telescope-card');
        const rect = card.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        if (window.animationController) {
            window.animationController.createParticleBurst(centerX, centerY, '#4ecdc4');
        }
    }
});
