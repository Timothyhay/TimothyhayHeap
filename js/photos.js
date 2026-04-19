document.addEventListener('DOMContentLoaded', () => {

    // --- 照片数据 ---
    const photoData = PHOTO_DATA_FROM_JEKYLL;


    // --- 核心变量 ---
    const contentArea = document.getElementById('contentArea');
    const container = document.querySelector('.container')
    const galleryTemplate = document.getElementById('galleryWindowTemplate');
    const viewerTemplate = document.getElementById('imageViewerTemplate');
    const copyrightEl = document.querySelector('.copyright');
    const sidebar = document.querySelector('.sidebar');
    let highestZIndex = 101; 

    // --- 初始化函数 ---
    function initPhotoApp() {
        if (!container || !galleryTemplate || !viewerTemplate) {
            console.error('Required elements for Photo App are missing!');
            return;
        }
        createGalleryWindow();
    }

    // --- 创建主照片库窗口 ---
    function createGalleryWindow() {
        const galleryClone = galleryTemplate.content.cloneNode(true);
        const galleryWindow = galleryClone.querySelector('.dialog-box');
        const photoGrid = galleryClone.querySelector('.photo-grid');
        const photoCountStatus = galleryClone.querySelector('#photo-count');
        const maximizeButton = galleryClone.querySelector('.dialog-maximize-button');
        const closeButton = galleryClone.querySelector('.dialog-close-button');

        // 填充缩略图
        photoData.forEach((photo, index) => {
            const thumbDiv = document.createElement('div');
            thumbDiv.className = 'photo-thumbnail';
            thumbDiv.style.animationDelay = `${index * 0.05}s`;

            const img = document.createElement('img');
            img.src = photo.thumbnailUrl;
            img.alt = photo.title;
            img.loading = 'lazy';

            thumbDiv.appendChild(img);

            thumbDiv.addEventListener('click', () => {
                createImageViewerWindow(photo);
            });

            photoGrid.appendChild(thumbDiv);
        });

        photoCountStatus.textContent = `${photoData.length} items`;

        maximizeButton.addEventListener('click', () => toggleMaximize(galleryWindow));
        closeButton.addEventListener('click', () => {
            galleryWindow.remove();
            document.body.classList.remove('window-maximized');
        });

        galleryWindow.style.left = '40px';
        galleryWindow.style.top = '120px'; 
        galleryWindow.style.zIndex = ++highestZIndex;
        makeDraggable(galleryWindow);

        container.appendChild(galleryWindow);
    }

    function createImageViewerWindow(photo) {
        const viewerClone = viewerTemplate.content.cloneNode(true);
        const viewerWindow = viewerClone.querySelector('.dialog-box');

        viewerWindow.querySelector('.dialog-title').textContent = `View: ${photo.title}`;
        viewerWindow.querySelector('.full-size-photo').src = photo.fullUrl;
        viewerWindow.querySelector('.full-size-photo').alt = photo.title;
        viewerWindow.querySelector('.info-filename').textContent = photo.title;
        viewerWindow.querySelector('.info-date').textContent = photo.date;
        viewerWindow.querySelector('.info-location').textContent = photo.location;

        const galleryWindow = document.querySelector('.photo-gallery-window');
        let offsetX = 100;
        let offsetY = 50;
        if (galleryWindow) {
            const rect = galleryWindow.getBoundingClientRect();
            const parentRect = container.getBoundingClientRect();
            offsetX = rect.left - parentRect.left + Math.random() * 150 + 50;
            offsetY = rect.top - parentRect.top + Math.random() * 100 + 20;
        }

        viewerWindow.style.left = `${offsetX}px`;
        viewerWindow.style.top = `${offsetY}px`;
        viewerWindow.style.zIndex = ++highestZIndex;

        viewerWindow.querySelector('.dialog-close-button').addEventListener('click', () => {
            viewerWindow.remove();
        });
        makeDraggable(viewerWindow);

        container.appendChild(viewerWindow);
    }


    function makeDraggable(element) {
        const titleBar = element.querySelector('.dialog-title-bar');
        if (!titleBar) return;

        let offsetX, offsetY, isDragging = false;

        titleBar.addEventListener('mousedown', (e) => {
            if (element.classList.contains('maximized') || e.target.closest('button')) {
                return;
            }
            isDragging = true;

            const rect = element.getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;

            element.style.zIndex = ++highestZIndex;
            titleBar.style.cursor = 'grabbing';

            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            e.preventDefault();

            const parentRect = container.getBoundingClientRect();
            let newX = e.clientX - offsetX - parentRect.left;
            let newY = e.clientY - offsetY - parentRect.top;

            newX = Math.max(0 - element.offsetWidth + 40, Math.min(newX, parentRect.width - 40));
            newY = Math.max(0, Math.min(newY, parentRect.height - titleBar.offsetHeight));

            element.style.left = `${newX}px`;
            element.style.top = `${newY}px`;
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                titleBar.style.cursor = 'grab';
            }
        });
    }

    function toggleMaximize(windowElement) {
        const titleBar = windowElement.querySelector('.dialog-title-bar');
        const maximizeButton = windowElement.querySelector('.dialog-maximize-button');
        const isMaximized = windowElement.classList.contains('maximized');

        if (isMaximized) {
            windowElement.style.width = windowElement.dataset.originalWidth;
            windowElement.style.height = windowElement.dataset.originalHeight;
            windowElement.style.top = windowElement.dataset.originalTop;
            windowElement.style.left = windowElement.dataset.originalLeft;

            windowElement.classList.remove('maximized');
            document.body.classList.remove('window-maximized');

            maximizeButton.textContent = '▢';
            titleBar.style.cursor = 'grab';

            if (copyrightEl && copyrightEl.dataset.originalHtml) {
                copyrightEl.innerHTML = copyrightEl.dataset.originalHtml;
            }

        } else {
            windowElement.dataset.originalWidth = windowElement.style.width || `${windowElement.offsetWidth}px`;
            windowElement.dataset.originalHeight = windowElement.style.height || `${windowElement.offsetHeight}px`;
            windowElement.dataset.originalTop = windowElement.style.top;
            windowElement.dataset.originalLeft = windowElement.style.left;

            // 修正：相对于 #contentArea 定位
            windowElement.style.width = `100%`;
            windowElement.style.height = `100%`;
            windowElement.style.top = `0px`;
            windowElement.style.left = `0px`;

            windowElement.style.transition = 'all 0.4s ease-in-out';
            windowElement.classList.add('maximized');
            document.body.classList.add('window-maximized');

            maximizeButton.textContent = '❐';
            titleBar.style.cursor = 'default';

            if (copyrightEl) {
                if (!copyrightEl.dataset.originalHtml) {
                    copyrightEl.dataset.originalHtml = copyrightEl.innerHTML;
                }
                copyrightEl.innerHTML = '© 2025 TangerineSoda. All Rights Reserved';
            }
        }
    }

    initPhotoApp();
});
