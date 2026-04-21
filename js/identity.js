// Wait for the HTML document to be fully loaded and parsed before executing the script
document.addEventListener('DOMContentLoaded', () => {
    // Get references to key DOM elements
    const bootScreen = document.getElementById('boot-screen');
    const dialogTemplate = document.getElementById('dialogTemplate');
    const container = document.querySelector('.container');

    // Initialize z-index counter for dialog stacking.
    // Set much higher to stay above normal app windows which usually start at 1000
    let highestZIndex = 5000;

    // Configuration for dialog spawning
    const MAX_DIALOGS_TO_SPAWN = 10;
    let spawnedDialogsCount = 0;
    let DIALOGS_DATA_FROM_JSON = [];
    let availableDialogIndices = [];
    let dialogCreationInterval;

    // --- Boot Sequence Handler ---
    function handleBootSequence() {
        const skipBoot = localStorage.getItem('sodaOS_skipBoot') === 'true';
        const crtOverlay = document.querySelector('.crt-overlay');
        const crtEnabled = localStorage.getItem('sodaOS_crtEnabled') !== 'false';

        // Apply CRT initial state
        if (crtOverlay) {
            crtOverlay.style.display = crtEnabled ? 'block' : 'none';
        }

        if (!bootScreen || skipBoot) {
            if (bootScreen) bootScreen.remove();
            startDialogSystem();
            animateSidebarItems();
            return;
        }

        console.log("%c SodaFridge OS v1.0.4 Initializing... ", "background: #000; color: #0f0; font-weight: bold;");
        
        // After the progress bar animation (approx 4.2s total), fade out the boot screen
        setTimeout(() => {
            bootScreen.classList.add('fade-out');
            setTimeout(() => {
                bootScreen.remove();
                startDialogSystem();
                animateSidebarItems();
            }, 1000);
        }, 4500);
    }

    // --- System Settings Dialog ---
    function openSystemDialog() {
        // Prevent multiple system dialogs
        const existingDialog = document.querySelector('.dialog-box.system-dialog-window');
        if (existingDialog) {
            existingDialog.style.zIndex = 9999; // Always absolute top
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            existingDialog.classList.add('active');
            return;
        }

        const skipBoot = localStorage.getItem('sodaOS_skipBoot') === 'true';
        const crtEnabled = localStorage.getItem('sodaOS_crtEnabled') !== 'false';
        const turboMode = localStorage.getItem('sodaOS_turboMode') === 'true';

        const content = `
            <div class="system-settings">
                <div class="settings-group">
                    <p style="margin-bottom: 8px; font-weight: bold; border-bottom: 1px solid #888;">Startup & Visuals</p>
                    <label class="win98-checkbox">
                        <input type="checkbox" id="setting-skip-boot" ${skipBoot ? 'checked' : ''}>
                        <span>Skip Boot Animation</span>
                    </label>
                    <label class="win98-checkbox">
                        <input type="checkbox" id="setting-crt" ${crtEnabled ? 'checked' : ''}>
                        <span>CRT Scanline Effect</span>
                    </label>
                    <label class="win98-checkbox">
                        <input type="checkbox" id="setting-turbo" ${turboMode ? 'checked' : ''}>
                        <span>Turbo Mode (Fast Spawning)</span>
                    </label>
                </div>
                <div class="settings-group" style="margin-top: 15px;">
                    <p style="margin-bottom: 8px; font-weight: bold; border-bottom: 1px solid #888;">System Actions</p>
                    <button class="dialog-action-button" id="btn-reboot" style="width: 100%; margin: 5px 0; font-weight: bold;">REBOOT SYSTEM</button>
                    <button class="dialog-action-button" id="btn-clear-data" style="width: 100%; margin: 5px 0; font-weight: bold;">RESET ALL DATA</button>
                </div>
                <div style="margin-top: 15px; font-size: 9px; color: #666; text-align: center;">
                    SodaFridge OS v1.0.4<br>Kernel: WebKit/Blink Hybrid
                </div>
            </div>
        `;

        const dialog = createDialog({
            title: '⚙ System Control Panel',
            content: content,
            width: 260
        });

        // Add a specific class to identify this dialog
        dialog.classList.add('system-dialog-window');
        dialog.style.zIndex = 9999; // Set to very high value initially

        // --- Center the dialog manually ---
        const containerRect = container.getBoundingClientRect();
        const dialogWidth = 260;
        const dialogHeight = 280; // Estimated height

        const centerX = (containerRect.width - dialogWidth) / 2;
        const centerY = (containerRect.height - dialogHeight) / 2;

        dialog.style.left = `${centerX}px`;
        dialog.style.top = `${centerY}px`;

        // Add event listeners for the settings
        const skipCheck = dialog.querySelector('#setting-skip-boot');
        const crtCheck = dialog.querySelector('#setting-crt');
        const turboCheck = dialog.querySelector('#setting-turbo');
        const rebootBtn = dialog.querySelector('#btn-reboot');
        const resetBtn = dialog.querySelector('#btn-clear-data');

        skipCheck.addEventListener('change', (e) => {
            localStorage.setItem('sodaOS_skipBoot', e.target.checked);
        });

        crtCheck.addEventListener('change', (e) => {
            localStorage.setItem('sodaOS_crtEnabled', e.target.checked);
            const overlay = document.querySelector('.crt-overlay');
            if (overlay) overlay.style.display = e.target.checked ? 'block' : 'none';
        });

        turboCheck.addEventListener('change', (e) => {
            localStorage.setItem('sodaOS_turboMode', e.target.checked);
            location.reload(); // Reload to apply turbo intervals
        });

        rebootBtn.addEventListener('click', () => {
            localStorage.setItem('sodaOS_skipBoot', 'false'); // Force boot screen once
            window.location.href = '/';
        });

        resetBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to wipe all local settings?')) {
                localStorage.clear();
                location.reload();
            }
        });
    }

    // Bind sidebar trigger
    const systemTrigger = document.getElementById('systemSettingsTrigger');
    if (systemTrigger) {
        systemTrigger.addEventListener('click', (e) => {
            e.preventDefault();
            openSystemDialog();
        });
    }

    // --- Function to animate sidebar navigation items sequentially ---
    function animateSidebarItems() {
        const navItems = document.querySelectorAll('.sidebar .nav-item');
        navItems.forEach((item, index) => {
            item.style.transitionDelay = `${index * 0.07}s`;
            item.classList.add('nav-item-visible');
        });
    }

    // --- Function to initialize and shuffle available dialog indices ---
    function initializeAvailableDialogs() {
        if (DIALOGS_DATA_FROM_JSON.length === 0) return;
        availableDialogIndices = DIALOGS_DATA_FROM_JSON.map((_, index) => index);
        for (let i = availableDialogIndices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [availableDialogIndices[i], availableDialogIndices[j]] = [availableDialogIndices[j], availableDialogIndices[i]];
        }
    }

    // --- Function to create and display a new dialog box ---
    function createDialog(data) {
        if (!dialogTemplate) return null;

        const dialogClone = dialogTemplate.content.cloneNode(true);
        const dialogBox = dialogClone.querySelector('.dialog-box');
        const titleElement = dialogBox.querySelector('.dialog-title');
        const contentElement = dialogBox.querySelector('.dialog-content');
        const closeButton = dialogBox.querySelector('.dialog-close-button');
        const actionButtons = dialogBox.querySelectorAll('.dialog-action-button');
        const yesButton = actionButtons[0];
        const noButton = actionButtons[1];

        titleElement.textContent = data.title || 'Untitled Dialog';
        contentElement.innerHTML = data.content || '<p>No content.</p>';

        let dialogWidth = parseInt(data.width) || 300;
        const viewportWidth = window.innerWidth;

        if (viewportWidth < 480) {
            dialogWidth = Math.min(dialogWidth, viewportWidth * 0.85, 200);
            dialogWidth = Math.max(dialogWidth, 120);
        } else if (viewportWidth < 768) {
            dialogWidth = Math.min(dialogWidth, viewportWidth * 0.9, 280);
            dialogWidth = Math.max(dialogWidth, 150);
        }
        dialogBox.style.width = `${dialogWidth}px`;

        const containerRect = container.getBoundingClientRect();
        const containerPaddingTop = parseFloat(getComputedStyle(container).paddingTop) || 90;
        const maxX = Math.max(5, containerRect.width - dialogWidth - 5);
        const maxY = Math.max(5, containerRect.height - containerPaddingTop - 150);

        dialogBox.style.left = `${5 + Math.random() * maxX}px`;
        dialogBox.style.top = `${containerPaddingTop + 5 + Math.random() * maxY}px`;

        makeDraggable(dialogBox);

        const closeDialog = () => {
            dialogBox.classList.add('closing');
            setTimeout(() => dialogBox.remove(), 200);
        };

        closeButton.addEventListener('click', closeDialog);
        dialogBox.addEventListener('mousedown', () => {
            // System dialogs should stay above everything
            if (dialogBox.classList.contains('system-dialog-window')) {
                dialogBox.style.zIndex = 10000;
            } else {
                dialogBox.style.zIndex = ++highestZIndex;
            }
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            dialogBox.classList.add('active');
        }, true);

        if (noButton) noButton.addEventListener('click', closeDialog);
        if (yesButton) {
            if (data.yesLink) {
                yesButton.addEventListener('click', () => window.open(data.yesLink, '_blank'));
            } else {
                yesButton.textContent = 'OK';
                if (noButton) noButton.style.display = 'none';
                yesButton.addEventListener('click', closeDialog);
            }
        }

        container.appendChild(dialogBox);
        spawnedDialogsCount++;
        return dialogBox;
    }

    function makeDraggable(element) {
        const titleBar = element.querySelector('.dialog-title-bar');
        let offsetX, offsetY, isDragging = false;

        titleBar.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('dialog-close-button')) return;
            isDragging = true;
            offsetX = e.clientX - element.getBoundingClientRect().left;
            offsetY = e.clientY - element.getBoundingClientRect().top;
            
            if (element.classList.contains('system-dialog-window')) {
                element.style.zIndex = 10000;
            } else {
                element.style.zIndex = ++highestZIndex;
            }
            
            titleBar.style.cursor = 'grabbing';
            document.querySelectorAll('.dialog-box.active').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const containerRect = container.getBoundingClientRect();
            let newStyleLeft = (e.clientX - offsetX) - containerRect.left;
            let newStyleTop = (e.clientY - offsetY) - containerRect.top;
            
            const logoHeight = (document.querySelector('.main-logo')?.offsetHeight || 0) + 10;
            newStyleTop = Math.max(newStyleTop, logoHeight);
            
            element.style.left = `${newStyleLeft}px`;
            element.style.top = `${newStyleTop}px`;
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            titleBar.style.cursor = 'grab';
        });
    }

    function spawnNextDialog() {
        if (spawnedDialogsCount >= MAX_DIALOGS_TO_SPAWN || availableDialogIndices.length === 0) {
            if (dialogCreationInterval) clearInterval(dialogCreationInterval);
            return;
        }
        const data = DIALOGS_DATA_FROM_JSON[availableDialogIndices.shift()];
        if (data) createDialog(data);
    }

    function startDialogSystem() {
        // Only spawn random dialogs on the root index page
        const isHomePage = window.location.pathname === '/' || window.location.pathname === '/index.html';
        if (!isHomePage) return;

        const turboMode = localStorage.getItem('sodaOS_turboMode') === 'true';
        const initialDelay = turboMode ? 400 : 1200;
        const interval = turboMode ? 800 : 2200;

        initializeAvailableDialogs();
        spawnNextDialog();
        setTimeout(() => spawnNextDialog(), initialDelay);
        dialogCreationInterval = setInterval(spawnNextDialog, interval);
    }

    fetch('data/dialogs/dialogs-data.json')
        .then(r => r.json())
        .then(data => {
            DIALOGS_DATA_FROM_JSON = data;
            handleBootSequence();
        })
        .catch(e => {
            console.error(e);
            handleBootSequence();
        });

    // Clock Logic
    function updateClock() {
        const now = new Date();
        const year = document.getElementById('clock-year');
        const week = document.getElementById('clock-week');
        const h = document.getElementById('clock-hours');
        const m = document.getElementById('clock-minutes');
        const s = document.getElementById('clock-seconds');
        const cr = document.getElementById('cosmic-ray-intensity');

        if (year) year.textContent = now.getFullYear();
        if (h) h.textContent = String(now.getHours()).padStart(2, '0');
        if (m) m.textContent = String(now.getMinutes()).padStart(2, '0');
        if (s) s.textContent = String(now.getSeconds()).padStart(2, '0');
        if (cr) cr.textContent = (Math.random() * 2).toFixed(2);
    }
    setInterval(updateClock, 1000);
    updateClock();
});
