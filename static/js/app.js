/* CNN Visualization Toolkit - JavaScript */

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Add loading states to buttons
    addButtonLoadingStates();

    // Initialize tooltips and popovers
    initializeBootstrapComponents();

    // Add smooth scrolling
    addSmoothScrolling();

    // Add keyboard navigation
    addKeyboardNavigation();

    // Initialize fade-in animations
    initializeFadeInAnimations();

    // Add performance monitoring
    monitorPerformance();

    console.log('🚀 CNN Visualization Toolkit initialized');
}

function addButtonLoadingStates() {
    const buttons = document.querySelectorAll('.btn');

    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Don't add loading state to navigation buttons
            if (this.closest('.navbar') || this.hasAttribute('data-no-loading')) {
                return;
            }

            // Add loading state
            this.classList.add('loading');
            this.disabled = true;

            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';

            // Remove loading state after delay (simulated async operation)
            setTimeout(() => {
                this.classList.remove('loading');
                this.disabled = false;
                this.innerHTML = originalText;
            }, 1500);
        });
    });
}

function initializeBootstrapComponents() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function addKeyboardNavigation() {
    document.addEventListener('keydown', function(e) {
        // ESC key to close modals or reset selections
        if (e.key === 'Escape') {
            resetAllSelections();
        }

        // Ctrl+H for help
        if (e.ctrlKey && e.key === 'h') {
            e.preventDefault();
            showHelpModal();
        }

        // Space bar for auto-demo (if available)
        if (e.key === ' ' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
            triggerAutoDemo();
        }
    });
}

function initializeFadeInAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe cards and major elements
    const elementsToAnimate = document.querySelectorAll('.card, .jumbotron, .layer-representation');
    elementsToAnimate.forEach(element => {
        observer.observe(element);
    });
}

function monitorPerformance() {
    // Basic performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(() => {
                const perfData = performance.getEntriesByType('navigation')[0];
                const loadTime = perfData.loadEventEnd - perfData.loadEventStart;

                console.log(`⚡ Page load time: ${loadTime}ms`);

                // Show performance badge if load time is good
                if (loadTime < 1000) {
                    showPerformanceBadge('Fast Load', 'success');
                } else if (loadTime < 3000) {
                    showPerformanceBadge('Good Load', 'warning');
                } else {
                    showPerformanceBadge('Slow Load', 'danger');
                }
            }, 0);
        });
    }
}

function showPerformanceBadge(text, type) {
    const badge = document.createElement('div');
    badge.className = `alert alert-${type} position-fixed bottom-0 end-0 m-3`;
    badge.style.zIndex = '9999';
    badge.style.opacity = '0';
    badge.innerHTML = `<i class="fas fa-tachometer-alt me-2"></i>${text}`;

    document.body.appendChild(badge);

    // Fade in
    setTimeout(() => {
        badge.style.transition = 'opacity 0.3s ease';
        badge.style.opacity = '1';
    }, 100);

    // Fade out after 3 seconds
    setTimeout(() => {
        badge.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(badge);
        }, 300);
    }, 3000);
}

function resetAllSelections() {
    // Remove active classes
    document.querySelectorAll('.active').forEach(element => {
        element.classList.remove('active');
    });

    // Reset any custom selections
    document.querySelectorAll('.selected').forEach(element => {
        element.classList.remove('selected');
    });

    console.log('🔄 All selections reset');
}

function showHelpModal() {
    // Create help modal if it doesn't exist
    if (!document.getElementById('helpModal')) {
        const modal = document.createElement('div');
        modal.id = 'helpModal';
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-question-circle me-2"></i>Help & Keyboard Shortcuts
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <h6>Keyboard Shortcuts:</h6>
                        <ul>
                            <li><kbd>Ctrl + H</kbd> - Show this help dialog</li>
                            <li><kbd>Esc</kbd> - Reset selections and close dialogs</li>
                            <li><kbd>Space</kbd> - Start auto-demo (where available)</li>
                            <li><kbd>Arrow Keys</kbd> - Navigate through layers (in architecture view)</li>
                        </ul>

                        <h6 class="mt-4">Navigation:</h6>
                        <ul>
                            <li><strong>Dashboard:</strong> Overview of the CNN visualization toolkit</li>
                            <li><strong>Architecture:</strong> Detailed view of the CNN model structure</li>
                            <li><strong>Feature Maps:</strong> Visualization of learned features</li>
                            <li><strong>Figure 1.6:</strong> Interactive recreation of the famous figure</li>
                            <li><strong>Operations:</strong> How convolution and pooling work</li>
                        </ul>

                        <h6 class="mt-4">Tips:</h6>
                        <ul>
                            <li>Click on layers to see detailed information</li>
                            <li>Hover over elements for additional context</li>
                            <li>Use the auto-demo to see animated walkthroughs</li>
                            <li>All visualizations are interactive and educational</li>
                        </ul>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // Show the modal
    const helpModal = new bootstrap.Modal(document.getElementById('helpModal'));
    helpModal.show();
}

function triggerAutoDemo() {
    // Look for auto-demo functionality on current page
    if (typeof startDemo === 'function') {
        startDemo();
        console.log('🎬 Auto-demo started');
    } else {
        console.log('ℹ️ No auto-demo available on this page');
    }
}

// Utility functions for API interactions
function fetchModelInfo() {
    return fetch('/api/model-info')
        .then(response => response.json())
        .catch(error => {
            console.error('Error fetching model info:', error);
            return null;
        });
}

function fetchFeatureMaps(imageName) {
    return fetch(`/api/feature-maps/${imageName}`)
        .then(response => response.json())
        .catch(error => {
            console.error('Error fetching feature maps:', error);
            return null;
        });
}

function fetchClassification(imageName) {
    return fetch(`/api/classification/${imageName}`)
        .then(response => response.json())
        .catch(error => {
            console.error('Error fetching classification:', error);
            return null;
        });
}

// Animation utilities
function animateElement(element, animation = 'pulse') {
    element.classList.add(animation);
    setTimeout(() => {
        element.classList.remove(animation);
    }, 1000);
}

function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
    notification.style.zIndex = '9999';
    notification.style.opacity = '0';
    notification.innerHTML = `
        <i class="fas fa-info-circle me-2"></i>
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;

    document.body.appendChild(notification);

    // Fade in
    setTimeout(() => {
        notification.style.transition = 'opacity 0.3s ease';
        notification.style.opacity = '1';
    }, 100);

    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.opacity = '0';
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.parentElement.removeChild(notification);
                }
            }, 300);
        }
    }, duration);
}

// Image loading utilities
function preloadImages(imageUrls) {
    const promises = imageUrls.map(url => {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = resolve;
            img.onerror = reject;
            img.src = url;
        });
    });

    return Promise.all(promises);
}

// Export functions for use in other scripts
window.CNNViz = {
    fetchModelInfo,
    fetchFeatureMaps,
    fetchClassification,
    animateElement,
    showNotification,
    preloadImages,
    resetAllSelections
};

console.log('📊 CNN Visualization Toolkit JavaScript loaded');
