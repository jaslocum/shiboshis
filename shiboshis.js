import fs from 'fs'
import fetch from 'node-fetch'
let shiboshis = []
fs.readFile('/Users/jack/shiboshis/shiboshis.json', 'utf8', function (err, data) {
	if (!err) {
		shiboshis = JSON.parse(data)
		process()
	} else {
		console.log(err)
	}
})
async function process() {
	let j = null
	for (let i in shiboshis) {
		j = Number(i) + 1
		j = String(j)
		console.log('down loading #' + j + ', image: ' + shiboshis[i].image + ' starting')
		await download(shiboshis[i].image, '/Users/jack/shiboshis/images/' + j + '.png')
	}
}
async function download (uri, filename) {
	let response = await fetch(uri)
	let buffer = await response.buffer()
	fs.writeFile(filename, buffer, () => {
			console.log(filename + 'finished downloading')
		}
	)
}
