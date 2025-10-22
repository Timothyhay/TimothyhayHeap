document.addEventListener('DOMContentLoaded', () => {

    // --- 照片数据 ---
    // 在这里定义你的照片信息。你可以根据需要添加更多。
    // 建议：为缩略图准备较小的版本以提高加载速度。
    const photoData = PHOTO_DATA_FROM_JEKYLL;


    // --- 核心变量 ---
    const contentArea = document.getElementById('contentArea');
    const container = document.querySelector('.container')
    const galleryTemplate = document.getElementById('galleryWindowTemplate');
    const viewerTemplate = document.getElementById('imageViewerTemplate');
    let highestZIndex = 101; // 与 identity.js 保持一致或更高

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
            thumbDiv.style.animationDelay = `${index * 0.05}s`; // 错开动画效果

            const img = document.createElement('img');
            img.src = photo.thumbnailUrl;
            img.alt = photo.title;
            img.loading = 'lazy'; // 懒加载图片

            thumbDiv.appendChild(img);

            // 点击缩略图时，创建图片查看器
            thumbDiv.addEventListener('click', () => {
                createImageViewerWindow(photo);
            });

            photoGrid.appendChild(thumbDiv);
        });

        // 更新状态栏
        photoCountStatus.textContent = `${photoData.length} items`;

        // 添加按钮事件监听器
        maximizeButton.addEventListener('click', () => toggleMaximize(galleryWindow));
        closeButton.addEventListener('click', () => galleryWindow.remove());
        maximizeButton.innerHTML = '❐';

        // 放置窗口并使其可拖动
        galleryWindow.style.left = '40px';
        galleryWindow.style.top = '120px'; // 放在 LOGO 下方
        galleryWindow.style.zIndex = ++highestZIndex;
        makeDraggable(galleryWindow);

        container.appendChild(galleryWindow);
    }

    // --- 创建单个图片查看器窗口 ---
    function createImageViewerWindow(photo) {
        const viewerClone = viewerTemplate.content.cloneNode(true);
        const viewerWindow = viewerClone.querySelector('.dialog-box');

        // 填充数据
        viewerWindow.querySelector('.dialog-title').textContent = `View: ${photo.title}`;
        viewerWindow.querySelector('.full-size-photo').src = photo.fullUrl;
        viewerWindow.querySelector('.full-size-photo').alt = photo.title;
        viewerWindow.querySelector('.info-filename').textContent = photo.title;
        viewerWindow.querySelector('.info-date').textContent = photo.date;
        viewerWindow.querySelector('.info-location').textContent = photo.location;

        // 随机放置新窗口，使其看起来更自然
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

        // 添加关闭和拖动功能
        viewerWindow.querySelector('.dialog-close-button').addEventListener('click', () => {
            viewerWindow.remove();
        });
        makeDraggable(viewerWindow);

        container.appendChild(viewerWindow);
    }


    // --- 可拖动窗口的辅助函数 - 可拖动区域与 indentity.js 不同 ---
    function makeDraggable(element) {
        const titleBar = element.querySelector('.dialog-title-bar');
        if (!titleBar) return;

        let offsetX, offsetY, isDragging = false;

        titleBar.addEventListener('mousedown', (e) => {
            // if (e.target.classList.contains('dialog-close-button')) return;
            if (element.dataset.isMaximized === 'true' || e.target.closest('button')) {
                return;
            }
            isDragging = true;

            const rect = element.getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;

            element.style.zIndex = ++highestZIndex;
            titleBar.style.cursor = 'grabbing';

            // 移除其他窗口的 active 状态
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            e.preventDefault();

            const parentRect = container.getBoundingClientRect();
            let newX = e.clientX - offsetX - parentRect.left;
            let newY = e.clientY - offsetY - parentRect.top;

            // 约束在 contentArea 内 (大致)
            const minX = 0 - element.offsetWidth + 40; // 允许部分移出
            const maxX = parentRect.width - 40;
            const minY = 0;
            const maxY = parentRect.height - titleBar.offsetHeight;

            newX = Math.max(minX, Math.min(newX, maxX));
            newY = Math.max(minY, Math.min(newY, maxY));

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

    // 最大化/还原窗口的函数
    function toggleMaximize(windowElement) {
        const titleBar = windowElement.querySelector('.dialog-title-bar');
        const maximizeButton = windowElement.querySelector('.dialog-maximize-button');
        const copyrightEl = document.querySelector('.copyright');
        const sidebar = document.querySelector('.sidebar');
        const mainLogo = document.querySelector('.main-logo')

        const isMaximized = windowElement.dataset.isMaximized === 'true';



        if (isMaximized) {
            // --- 还原窗口 ---
            windowElement.style.width = windowElement.dataset.originalWidth;
            windowElement.style.height = windowElement.dataset.originalHeight;
            windowElement.style.top = windowElement.dataset.originalTop;
            windowElement.style.left = windowElement.dataset.originalLeft;

            windowElement.dataset.isMaximized = 'false';
            windowElement.classList.remove('maximized');

            maximizeButton.innerHTML = '❐';
            maximizeButton.title = 'Maximize';
            titleBar.style.cursor = 'grab';

            // 移除全局状态并恢复 copyright 内容
            document.body.classList.remove('gallery-maximized');
            if (copyrightEl && copyrightEl.dataset.originalHtml) {
                copyrightEl.innerHTML = copyrightEl.dataset.originalHtml;
            }

        } else {
            // --- 最大化窗口 ---
            windowElement.dataset.originalWidth = windowElement.style.width || `${windowElement.offsetWidth}px`;
            windowElement.dataset.originalHeight = windowElement.style.height || `${windowElement.offsetHeight}px`;
            windowElement.dataset.originalTop = windowElement.style.top;
            windowElement.dataset.originalLeft = windowElement.style.left;

            // 智能最大化计算
            // 检查 sidebar 是否可见且为侧边栏模式 (非垂直堆叠)
            // getComputedStyle(sidebar).display !== 'none' 检查它是否被隐藏
            // window.innerWidth > 768 是一个简单的断点检查，与你的CSS响应式断点对应
            let sidebarWidth = 0;
            if (sidebar && getComputedStyle(sidebar).display !== 'none' && window.innerWidth > 768) {
                sidebarWidth = sidebar.offsetWidth;
            }

            let logoTopOffset = 0;
            if (mainLogo) {
                const logoRect = mainLogo.getBoundingClientRect();
                const containerRect = mainLogo.parentElement.getBoundingClientRect(); // 获取父容器(.container)的 rect
                // 计算 LOGO 底部相对于父容器顶部的距离，并增加一些间距
                logoTopOffset = (logoRect.bottom - containerRect.top) - 20; // 2px 的额外间距
            }

            // 应用最大化样式，并留出 sidebar & logo 的空间
            windowElement.style.width = `calc(100% - ${sidebarWidth}px)`;
            windowElement.style.height = `calc(100% - ${logoTopOffset}px)`; // 使用 calc() 调整高度
            windowElement.style.top = `${logoTopOffset}px`; // 从 LOGO 下方开始
            windowElement.style.left = `${sidebarWidth}px`;

            // 为最大化/还原添加平滑过渡动画
            windowElement.style.transition = 'width 0.3s ease-out, height 0.3s ease-out, top 0.3s ease-out, left 0.3s ease-out';

            // 设置状态
            windowElement.dataset.isMaximized = 'true';
            windowElement.classList.add('maximized');

            maximizeButton.innerHTML = '□';
            maximizeButton.title = 'Restore';
            titleBar.style.cursor = 'default';

            // 添加全局状态并修改 copyright 内容
            document.body.classList.add('gallery-maximized');
            if (copyrightEl) {
                // 首次最大化时，保存原始内容
                if (!copyrightEl.dataset.originalHtml) {
                    copyrightEl.dataset.originalHtml = copyrightEl.innerHTML;
                }
                // 修改为精简内容
                copyrightEl.innerHTML = '© 2025 TangerineSoda. All Rights Reserved';
            }
        }
    }

    // --- 启动照片应用 ---
    initPhotoApp();
});