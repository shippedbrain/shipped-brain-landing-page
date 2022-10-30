function getMarkdown() {
    fetch(`../assets/quickstart.md`)
        .then(response => response.text())
        .then(result => document.getElementById('quickstart-container').innerHTML = marked(result))
}

getMarkdown()