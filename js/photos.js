document.addEventListener('DOMContentLoaded', () => {
    const photoManagerWindow = document.querySelector('.photo-manager-window');
    const contentArea = document.getElementById('contentArea');
    const photoGrid = document.getElementById('photo-grid');
    const photoSearch = document.getElementById('photo-search');
    const statusTotal = document.getElementById('status-total');
    const statusSelected = document.getElementById('status-selected');
    const selectionInfo = document.getElementById('selection-info');
    const sidebarItems = document.querySelectorAll('.explorer-sidebar .sidebar-item');
    
    // Viewer elements
    const viewerModal = document.getElementById('image-viewer-modal');
    const viewerImg = document.getElementById('viewer-img');
    const viewerTitle = document.getElementById('viewer-title');
    const viewerMaxBtn = document.getElementById('viewer-maximize-btn');

    let highestZIndex = 1000;
    let currentCategory = 'ALL';
    let selectedItem = null;

    // --- Window Dragging ---
    function makeDraggable(element) {
        const titleBar = element.querySelector('.dialog-title-bar');
        if (!titleBar) return;
        let offsetX, offsetY, isDragging = false;

        titleBar.addEventListener('mousedown', (e) => {
            if (element.classList.contains('maximized') || e.target.closest('button')) return;
            isDragging = true;
            const rect = element.getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;
            element.style.zIndex = ++highestZIndex;
            titleBar.style.cursor = 'grabbing';
            element.style.transition = 'none';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const areaRect = contentArea.getBoundingClientRect();
            let newX = (e.clientX - offsetX) - areaRect.left;
            let newY = (e.clientY - offsetY) - areaRect.top;
            newY = Math.max(-80, newY); 
            element.style.left = `${newX}px`;
            element.style.top = `${newY}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                titleBar.style.cursor = 'grab';
                element.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out, width 0.3s ease-out, height 0.3s ease-out, top 0.3s ease-out, left 0.3s ease-out';
            }
        });
    }

    // --- Maximize Logic (Shared/Adapted) ---
    function toggleMaximize(windowElement, isViewer = false) {
        const titleBar = windowElement.querySelector('.dialog-title-bar');
        const maximizeButton = windowElement.querySelector('.dialog-maximize-button');
        const isMaximized = windowElement.classList.contains('maximized');
        const areaRect = contentArea.getBoundingClientRect();

        if (isMaximized) {
            windowElement.style.width = windowElement.dataset.originalWidth;
            windowElement.style.height = windowElement.dataset.originalHeight;
            windowElement.style.top = windowElement.dataset.originalTop;
            windowElement.style.left = windowElement.dataset.originalLeft;
            windowElement.classList.remove('maximized');
            maximizeButton.textContent = '▢';
            if (!isViewer) titleBar.style.cursor = 'grab';
        } else {
            windowElement.dataset.originalWidth = windowElement.style.width || `${windowElement.offsetWidth}px`;
            windowElement.dataset.originalHeight = windowElement.style.height || `${windowElement.offsetHeight}px`;
            windowElement.dataset.originalTop = windowElement.style.top;
            windowElement.dataset.originalLeft = windowElement.style.left;

            if (isViewer) {
                // Viewer maximizes to 95% of area
                const mw = areaRect.width * 0.95;
                const mh = areaRect.height * 0.95;
                windowElement.style.width = `${mw}px`;
                windowElement.style.height = `${mh}px`;
                windowElement.style.left = `${(areaRect.width - mw) / 2}px`;
                windowElement.style.top = `${(areaRect.height - mh) / 2}px`;
            } else {
                windowElement.style.width = `100%`;
                windowElement.style.height = `100%`;
                windowElement.style.top = `0px`;
                windowElement.style.left = `0px`;
                titleBar.style.cursor = 'default';
            }

            windowElement.classList.add('maximized');
            maximizeButton.textContent = '❐';
        }
    }

    // --- Photo Selection & Filtering ---
    window.filterByCategory = function(category) {
        currentCategory = category;
        sidebarItems.forEach(item => {
            const text = item.textContent.toUpperCase();
            if (text.includes(category) || (category === 'ALL' && text.includes('ALL'))) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        applyFilters();
    };

    window.searchPhotos = function() {
        applyFilters();
    };

    function applyFilters() {
        const query = photoSearch.value.toUpperCase();
        const items = document.querySelectorAll('.photo-item');
        let visibleCount = 0;

        items.forEach(item => {
            const cat = item.getAttribute('data-category');
            const title = item.getAttribute('data-title').toUpperCase();
            const matchesCat = currentCategory === 'ALL' || cat === currentCategory;
            const matchesQuery = title.includes(query);

            if (matchesCat && matchesQuery) {
                item.style.display = 'flex';
                visibleCount++;
                const img = item.querySelector('img');
                if (img && img.dataset.src && !img.src) {
                    img.src = img.dataset.src;
                }
            } else {
                item.style.display = 'none';
            }
        });
        statusTotal.textContent = `Total: ${visibleCount} Items`;
    }

    function selectPhoto(item) {
        if (selectedItem) selectedItem.classList.remove('selected');
        selectedItem = item;
        item.classList.add('selected');
        
        const title = item.getAttribute('data-title');
        const cat = item.getAttribute('data-category');
        selectionInfo.innerHTML = `
            <strong>File:</strong> ${title}<br>
            <strong>Type:</strong> JPEG Image<br>
            <strong>Folder:</strong> ${cat}
        `;
        statusSelected.textContent = `Selected: 1`;
    }

    // --- Image Viewer & Zoom ---
    window.openImageViewer = function(src, title) {
        viewerImg.classList.remove('zoomed');
        viewerImg.src = src;
        viewerTitle.textContent = `Image Viewer - ${title}`;
        viewerModal.style.display = 'flex';
        viewerModal.style.zIndex = ++highestZIndex;
        viewerModal.classList.remove('maximized');
        viewerMaxBtn.textContent = '▢';
        
        const areaRect = contentArea.getBoundingClientRect();
        const defaultWidth = Math.min(areaRect.width * 0.8, 800);
        const defaultHeight = Math.min(areaRect.height * 0.8, 600);
        
        viewerModal.style.width = `${defaultWidth}px`;
        viewerModal.style.height = `${defaultHeight}px`;
        viewerModal.style.left = `${(areaRect.width - defaultWidth) / 2}px`;
        viewerModal.style.top = `${(areaRect.height - defaultHeight) / 2}px`;
        
        if (!viewerModal.dataset.draggable) {
            makeDraggable(viewerModal);
            viewerModal.dataset.draggable = 'true';
        }
    };

    if (viewerMaxBtn) {
        viewerMaxBtn.addEventListener('click', () => toggleMaximize(viewerModal, true));
    }

    window.closeImageViewer = function() {
        viewerModal.style.display = 'none';
    };

    window.toggleZoom = function(img) {
        const container = document.getElementById('viewer-container');
        if (img.classList.contains('zoomed')) {
            img.classList.remove('zoomed');
            container.style.alignItems = 'center';
            container.style.justifyContent = 'center';
        } else {
            img.classList.add('zoomed');
            container.style.alignItems = 'flex-start';
            container.style.justifyContent = 'flex-start';
        }
    };

    // --- Lazy Loading ---
    const photoObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    photoObserver.unobserve(img);
                }
            }
        });
    }, { root: photoGrid, threshold: 0.1 });

    // --- Global Sidebar Fix ---
    function ensureSidebarVisible() {
        const navItems = document.querySelectorAll('.sidebar .nav-item');
        if (navItems.length > 0 && !navItems[0].classList.contains('nav-item-visible')) {
            navItems.forEach((item, index) => {
                item.style.transitionDelay = `${index * 0.07}s`;
                item.classList.add('nav-item-visible');
            });
        }
    }

    // --- Init ---
    if (photoManagerWindow) {
        ensureSidebarVisible();
        makeDraggable(photoManagerWindow);
        const maxBtn = photoManagerWindow.querySelector('.dialog-maximize-button');
        if (maxBtn) maxBtn.addEventListener('click', () => toggleMaximize(photoManagerWindow));
        
        const closeBtn = photoManagerWindow.querySelector('.dialog-close-button');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                photoManagerWindow.style.opacity = '0';
                setTimeout(() => window.location.href = '/', 300);
            });
        }

        document.querySelectorAll('.photo-item').forEach(item => {
            const img = item.querySelector('img');
            if (img) photoObserver.observe(img);

            item.addEventListener('click', (e) => selectPhoto(item));
            item.addEventListener('dblclick', () => {
                const src = item.getAttribute('data-src');
                const title = item.getAttribute('data-title');
                openImageViewer(src, title);
            });
        });

        const areaRect = contentArea.getBoundingClientRect();
        setTimeout(() => {
            photoManagerWindow.style.left = `${(areaRect.width - photoManagerWindow.offsetWidth) / 2}px`;
            photoManagerWindow.style.top = `30px`; 
            photoManagerWindow.style.opacity = '1';
            photoManagerWindow.style.transform = 'scale(1)';
            applyFilters();
        }, 100);
    }
});
