// Add CSRF token to all AJAX requests
document.addEventListener('DOMContentLoaded', function() {
    // Function to get CSRF token, with fallback to fetching it from server
    function getCSRFToken() {
        // Try to get from meta tag first
        const metaTag = document.querySelector('meta[name="csrf-token"]');
        if (metaTag && metaTag.getAttribute('content')) {
            return metaTag.getAttribute('content');
        }
        
        // If not available, fetch from server
        return fetch('/get-csrf-token')
            .then(response => response.json())
            .then(data => {
                return data.csrf_token;
            })
            .catch(error => {
                console.error('Error fetching CSRF token:', error);
                return '';
            });
    }
    
    // Get initial token
    const tokenPromise = getCSRFToken();
    
    // Handle jQuery AJAX if available
    if (typeof jQuery !== 'undefined') {
        jQuery(document).ajaxSend(function(e, xhr, options) {
            // Get the token and set it in the header
            if (typeof tokenPromise === 'string') {
                xhr.setRequestHeader("X-CSRFToken", tokenPromise);
            } else {
                tokenPromise.then(token => {
                    xhr.setRequestHeader("X-CSRFToken", token);
                });
            }
        });
    }
    
    // Override fetch to include CSRF token
    const originalFetch = window.fetch;
    window.fetch = function(url, options = {}) {
        // Only add token for API calls and non-GET requests
        if (typeof url === 'string' && 
            (url.startsWith('/api/') || 
             (options.method && options.method.toUpperCase() !== 'GET'))) {
            
            return Promise.resolve(tokenPromise)
                .then(token => {
                    options = options || {};
                    options.headers = options.headers || {};
                    
                    // Only add token if not already present
                    if (!options.headers['X-CSRFToken']) {
                        options.headers['X-CSRFToken'] = token;
                    }
                    
                    return originalFetch(url, options);
                });
        }
        
        // Pass through for non-API or GET requests
        return originalFetch(url, options);
    };
}); 