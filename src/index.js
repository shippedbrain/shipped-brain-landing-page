import PR from '../dist/js/prettify'

let navMenuToggle = document.getElementById('navMenuToggle')
let navMenuCloseBtn = document.getElementById('navOverlayCloseBtn')
let navFullscreen = document.getElementById('navbarFullScreen')
let navIntegrationLink = document.getElementById('navIntegrationLink')

/**
 * Method to handle tab switching
 */
function toggleTab() {
    let tabToggles = Array.from(document.getElementsByClassName('tab-toggle'))

    tabToggles.forEach(toggle => {
        toggle.addEventListener('click', () => {
            let dataTarget = toggle.getAttribute('data-target')
            let tabHeader = document.getElementById(toggle.getAttribute('data-tab-header'))
            let tabContentGroup = document.getElementById(toggle.getAttribute('data-content-group'))
            let type = toggle.getAttribute('data-type')
            let targetContent = document.getElementById(dataTarget)

            Array.from(tabHeader.children).forEach(tabPill => {
                removeActiveFromElement(tabPill.id)
            })
            Array.from(tabContentGroup.children).forEach(tabContent => {
                removeActiveFromElement(tabContent.id)
            })

            toggle.classList.add('active')
            targetContent.classList.add('active')

            switch (type) {
                case 'code':
                    targetContent.innerHTML = getCodeExample(dataTarget)
                    break
            }

            PR.prettyPrint()
        })
    })
}

function removeActiveFromElement(elementID) {
    let element = document.getElementById(elementID)
    element.classList.remove('active')
}

function getCodeExample(target) {
    switch (target) {
        case 'pythonExample':
            return getPythonExample()
        case 'javascriptExample':
            return getJavaScriptExample()
        default:
            return getPythonExample()
    }
}

function getPythonExample() {
    return `<pre class="prettyprint linenums">
import requests

url = "https://shippedbrain.com/api/v0/predict/sentiment-analysis/1"

data = {
    "columns": [
        "text"
    ],
    "data": [
        ["I'm so happy"],
        ["not so good"]
    ]
}

def sentiment_analysis(data):
    results = requests.post(url, data)
    return results
    </pre>`
}

function getJavaScriptExample() {
    return `<pre class="prettyprint linenums">
const data = {
    columns: [
        "text"
    ],
    data: [
        [ "I'm so happy" ],
        [ "not so good" ]
    ]
}

async function getSentimentAnalysis(data) {
    const url = 'https://shippedbrain.com/api/v0/predict/sentiment-analysis/1'
    const request = await fetch(url, { method: 'POST', body: data })
    const results = await request.json()

    return results
}

getSentimentAnalysis(data)
    </pre>`
}

function setCodeExample() {
    let toggle = document.getElementById('pythonTabToggle')
    let dataTarget = 'pythonExample'
    let targetContent = document.getElementById(dataTarget)

    toggle.classList.add('active')
    targetContent.classList.add('active')

    targetContent.innerHTML = getCodeExample(dataTarget)

    PR.prettyPrint()
}

function setCurrentYearOnFooter() {
    let footerYearElement = document.getElementById('footerYear')
    const year = new Date().getFullYear();

    footerYearElement.innerText = year
}

function removeAnchorLinkFromURLOnInit() {
    if (location.hash != '') {
        location.hash = ''
    
        history.pushState('', document.title, location.href.replace(/#.*/, ''))
    }    
}

function toggleFullscreenNav(isOpen) {
    navFullscreen.style.width = isOpen ? '100%' : '0%'
}

navMenuToggle.addEventListener('click', () => {
    toggleFullscreenNav(true)
})

navMenuCloseBtn.addEventListener('click', () => {
    toggleFullscreenNav(false)
})

navIntegrationLink.addEventListener('click', () => {
    toggleFullscreenNav(false)
})

removeAnchorLinkFromURLOnInit()
toggleTab()
setCodeExample()
setCurrentYearOnFooter()