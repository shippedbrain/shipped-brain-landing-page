const path = require('path')

module.exports = {
    "mode": "none",
    "entry": ["babel-polyfill", "./src/index.js"],
    "output": {
        "path": __dirname + '/dist',
        "filename": "bundle.js"
    },
    devServer: {
        contentBase: path.join(__dirname, 'dist'),
        open: true,
        overlay: true,
        watchContentBase: true,
        liveReload: true
    },
    "module": {
        "rules": [
            {
                "test": /\.css$/,
                "use": [
                    "style-loader",
                    "css-loader"
                ]
            },
            {
                "test": /\.js$/,
                "exclude": /node_modules/,
                "use": {
                    "loader": "babel-loader",
                    "options": {
                        "presets": [
                            "@babel/preset-env"
                        ]
                    }
                }
            }
        ]
    }
}