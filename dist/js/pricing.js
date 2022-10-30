let navMenuToggle = document.getElementById('navMenuToggle')
let navMenuCloseBtn = document.getElementById('navOverlayCloseBtn')
let navFullscreen = document.getElementById('navbarFullScreen')

navMenuToggle.addEventListener('click', () => {
    navFullscreen.style.width = '100%'
})

navMenuCloseBtn.addEventListener('click', () => {
    navFullscreen.style.width = '0%'
})