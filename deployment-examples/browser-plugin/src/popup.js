/**
 * @fileoverview Description of this file.
 */
let changeColorEl = document.getElementById('changeColor');

chrome.storage.sync.get('color', function(data) {
  changeColorEl.style.backgroundColor = data.color;
  changeColorEl.setAttribute('value', data.color);
});

changeColorEl.onclick = function(element) {
  let color = element.target.value;
  chrome.tabs.query({
    active: true,
    currentWindow: true},
    function(tabs) {
      chrome.tabs.executeScript(
        tabs[0].id,
        {code: 'document.body.style.backgroundColor = "' + color + '";'});
    }
  );
};
