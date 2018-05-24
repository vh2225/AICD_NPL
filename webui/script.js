
const api = 'http://localhost:8000/intent';


function doAnnotate(text) {
    var displacy = new displaCyENT(api,{
        container: '#displacy_models',
        format: 'spacy',
        distance: 300,
        offsetX: 100
    });

    var res_data = postData(text, api, displacy);

}

function getRandomColor(){
    var o = Math.round, r = Math.random, s = 255;
    var r_1 = o(r()*s);
    var r_2 = o(r()*s);
    var r_3 = o(r()*s);
    var rgb = 'rgb(' + r_1 + ',' + r_2 + ',' + r_3 + ')';
    var rgba = 'rgba(' + r_1 + ',' + r_2 + ',' + r_3 + ', 0.2)';
    return [rgb, rgba];
}

function addColorToAnnotationSet(annotation_set){
    dataLength = annotation_set.length
    for (var i = 0; i < dataLength; i++) {
        var colors = getRandomColor()
        var rgb_color = colors[0]
        var rgba_color = colors[1]
        var data1 = '[data-entity][data-entity=' + annotation_set[i] + '] {background: ' + rgba_color +'; border-color: ' + rgb_color + ';}'
        var data2 ='[data-entity][data-entity=' + annotation_set[i] + ']::after {background: ' + rgb_color + ';}'
        document.styleSheets[0].insertRule(data1);
        document.styleSheets[0].insertRule(data2);
    }
}

function postData(text, url, displacy) {
  fetch(url, {
    body: JSON.stringify({"text": text, "id":1}), // must match 'Content-Type' header
    cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
    credentials: 'same-origin', // include, *omit
    headers: {
      'Access-Control-Allow-Origin': '*',
      'content-type': 'application/json',
      'Access-Control-Allow-Headers': '*',
      'Access-Control-Allow-Methods': '*',
      'user-agent': 'Mozilla/4.0 MDN Example'

    },
    method: 'POST', // *GET, PUT, DELETE, etc.
    mode: 'cors', // no-cors, *same-origin
    redirect: 'follow', // *manual, error
    referrer: 'no-referrer', // *client
  })
  .then(response => response.json())
  .then(data => {
    var docData = data;
    console.log(docData);
    addColorToAnnotationSet(docData['annotation_set']);
    displacy.render(docData['text'], docData['ents'], docData['annotation_set']);
//    var intent_div = document.getElementById('displacy_intent');
//    var textnode = document.createTextNode('Intent: ' + docData['intent']);
    $("#displacy_intent").html('Intent: ' + docData['intent'])
//    if (intent_div.hasChildNodes()) {
//        intent_div.removeChild(intent_div.childNodes);
//    }
//    intent_div.appendChild(textnode)

  }) // JSON from `response.json()` call
  .catch(error => {
     console.error(error)
  })
}
