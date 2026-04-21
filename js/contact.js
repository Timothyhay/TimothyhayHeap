let highestZIndex = 2000;

function makeDraggable(element) {
    const titleBar = element.querySelector('.dialog-title-bar');
    let offsetX, offsetY, isDragging = false;

    titleBar.addEventListener('mousedown', (e) => {
        if (e.target.closest('button')) return;
        isDragging = true;
        element.style.transition = 'none';
        element.style.transform = 'none';
        const rect = element.getBoundingClientRect();
        offsetX = e.clientX - rect.left;
        offsetY = e.clientY - rect.top;
        element.style.zIndex = ++highestZIndex;
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const parentRect = (element.offsetParent || document.body).getBoundingClientRect();
        let newX = (e.clientX - offsetX) - parentRect.left;
        let newY = (e.clientY - offsetY) - parentRect.top;
        newY = Math.max(10, newY);
        element.style.left = `${newX}px`;
        element.style.top = `${newY}px`;
    });

    document.addEventListener('mouseup', () => { isDragging = false; });
}

function ensureSidebarVisible() {
    const navItems = document.querySelectorAll('.sidebar .nav-item');
    if (navItems.length > 0 && !navItems[0].classList.contains('nav-item-visible')) {
        navItems.forEach((item, index) => {
            item.style.transitionDelay = `${index * 0.07}s`;
            item.classList.add('nav-item-visible');
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const win = document.getElementById('contactWindow');
    makeDraggable(win);
    ensureSidebarVisible();

    setTimeout(() => {
        const parent = win.offsetParent || document.body;
        const parentRect = parent.getBoundingClientRect();
        const centerX = (window.innerWidth - win.offsetWidth) / 2 - parentRect.left;
        const centerY = (window.innerHeight - win.offsetHeight) / 2 - parentRect.top;
        
        win.style.transition = 'none';
        win.style.left = `${centerX}px`;
        win.style.top = `${centerY}px`;
        win.style.transform = 'scale(0.95)';
        win.offsetHeight;
        
        win.style.transition = 'opacity 0.4s ease-out, transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
        win.style.opacity = '1';
        win.style.transform = 'scale(1)';
    }, 100);
});