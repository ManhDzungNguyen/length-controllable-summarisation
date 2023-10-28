const mainDiv = document.getElementsByClassName("main")[0]
const responseDiv = document.getElementsByClassName("responseBody")[0]

const settingsBtn = document.getElementById("settingsBtn")
const settings = document.getElementsByClassName("settings")[0]

const summarizationBtn = document.getElementById("summarizationBtn");
const responseTag = document.getElementById("response")


settingsBtn.addEventListener("click", () => {
	if (mainDiv.style.display !== "none") {
		mainDiv.style.display = "none"

		responseDiv.style.visibility = "hidden"
		responseDiv.style.display = "none"

		settingsBtn.innerHTML = "back"
		settings.style.visibility = "visible"
		settings.style.display = "flex"
	} else {
		mainDiv.style.display = "flex"

		// If response exists show responseDiv
		if (responseTag.innerHTML !== "") {
			responseDiv.style.visibility = "visible"
			responseDiv.style.display = "flex"
		}

		settingsBtn.innerHTML = "settings"
		settings.style.visibility = "hidden"
		settings.style.display = "none"
	}
});

// Save settings
const saveBtn = document.getElementById("saveBtn")
const noSentence = document.querySelector('input[name="nosentence"]')
const noToken = document.querySelector('input[name="notoken"]')


saveBtn.addEventListener("click", () => {
	const noSumSentence = parseInt(noSentence.value)
	const noSumToken = parseInt(noToken.value)
	chrome.storage.local.set({ nosentence: noSumSentence })
	chrome.storage.local.set({ notoken: noSumToken })
})

// Get nosentence from storage and fill the input
chrome.storage.local.get("nosentence", (data) => {
	const nosentence = data.nosentence;
	noSentence.value = nosentence || 5;
});
// Get notoken from storage and fill the input
chrome.storage.local.get("notoken", (data) => {
	const notoken = data.notoken;
	noToken.value = notoken || 500;
});


summarizationBtn.addEventListener("click", sendCurrentPageURL);

// // RESPONSE BODY
// // Load old response if it exists
// chrome.storage.local.get("response_content", (data) => {
// 	const response_content = data.response_content;
// 	if (response_content) {
// 		responseDiv.style.visibility = "visible"
// 		responseDiv.style.display = "flex"
// 		responseTag.innerHTML = response_content
// 	}
// });

async function sendCurrentPageURL() {
	try {
		responseTag.innerHTML = "Fetching response ..."
		responseDiv.style.visibility = "visible"
		responseDiv.style.display = "flex"
		// Get the current tab information.
		const queryOptions = { active: true, currentWindow: true };
		const [currentTab] = await chrome.tabs.query(queryOptions);


		const serverUrl = 'http://127.0.0.1:6868/text_extraction';

		const currentPageURL = currentTab.url;
		const noSentences = parseInt(noSentence.value);
		const noTokens = parseInt(noToken.value);

		// Create the request headers and body.
		const requestOptions = {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ url: currentPageURL, no_sentences: noSentences, no_tokens: noTokens }),
		};

		// Send the request to the server.
		const response = await fetch(serverUrl, requestOptions);

		if (!response.ok) {
			throw new Error(`HTTP error! Status: ${response.status}`);
		}

		const responseData = await response.json();

		if (responseData) {
			const formattedText = responseData["result"];
			responseTag.innerHTML = formattedText
			// Store response
			// chrome.storage.local.set({ response_content: formattedText })
		}


	} catch (error) {
		// Handle any errors that occur during the request.
		console.error('Failed to send the request:', error);
	}
}
