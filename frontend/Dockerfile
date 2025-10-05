# Use nginx alpine image
FROM nginx:alpine

# Copy HTML files
COPY index.html /usr/share/nginx/html/
COPY landing.html /usr/share/nginx/html/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
