importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter");

addEventListener("message", ({ data }) => {
  prefetch(data.page);
});

const ImageCache = "image-cache";

self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.open(ImageCache).then((cache) => {
      return cache.match(event.request).then((response) => {
        return (
          response ||
          fetch(event.request).then((response) => {
            return response;
          })
        );
      });
    })
  );
});

const MODEL_URL = "/assets/model.json";

let model = null;
tf.loadGraphModel(MODEL_URL).then((m) => (model = m));

const predict = async (path, userId) => {
  if (!model) {
    return;
  }
  const page = pages.indexOf(path);
  const pageId = tf.tensor1d([parseInt(page)], "int32");

  const sessionIndex = tf.tensor1d([parseInt(userId)], "int32");

  const result = model.predict({
    cur_page: pageId,
    session_index: sessionIndex,
  });
  const values = result.dataSync();
  const orders = sortWithIndeces(values).slice(0, 5);
  return orders;
};

const sortWithIndeces = (toSort) => {
  const sorted = [];
  for (var i = 0; i < toSort.length; i++) {
    sorted[i] = [toSort[i], pages[i]];
  }
  return sorted.sort((left, right) => {
    return left[0] > right[0] ? -1 : 1;
  });
};

const connectionSpeeds = { "slow-2g": 0.001, "2g": 0.00001, "3g": 0, "4g": 0 };

const prefetch = async (path, sessionId) => {
  const predictions = await predict(path, sessionId);
  const formattedPredictions = predictions
    .map(([a, b]) => `'${b}' -> ${a}`)
    .join("\n");
  console.log(`Navigating from: '${path}'`);
  console.log(formattedPredictions);
  const connectionSpeed = navigator.connection.effectiveType;
  const threshold = connectionSpeeds[connectionSpeed];
  const cache = await caches.open(ImageCache);
  predictions.forEach(async ([probability, category]) => {
    if (probability >= threshold) {
      const merchs = (await getMerchList(category)).map(getUrl);
      [...new Set(merchs)].forEach((url) => {
        const request = new Request(url, {
          mode: "no-cors",
        });
        fetch(request).then((response) => cache.put(request, response));
      });
    }
  });
};

const getMerchList = (category) => {
  return fetch(
    "http://localhost:8000/api/merch/" + ((category || "").trim() || "all")
  ).then((response) => response.json());
};

const getUrl = (merch) => {
  if (!merch) return undefined;
  return `https://firebasestorage.googleapis.com/v0/b/merch-store-daa40.appspot.com/o/${merch.id}.webp?alt=media`;
};

const pages = [
  "store.html",
  "",
  "quickview",
  "apparel-unisex",
  "google+redesign-clearance",
  "apparel",
  "new",
  "lifestyle-bags",
  "basket.html",
  "campus",
  "brand-youtube",
  "brand-google",
  "lifestyle-drinkware",
  "apparel-womens",
  "lifestyle",
  "apparel-unisex",
  "lifestyle-small",
  "signin.html",
  "yourinfo.html",
  "google+redesign-apparel-hats",
  "apparel-unisex",
  "asearch.html",
  "apparel-kids",
  "stationery-notebooks",
  "sale",
  "stationery-stickers",
  "men's t-shirts--quickview",
  "stationery",
  "brand-android",
  "apparel-socks",
  "stationery-writing",
  "payment.html",
  "myaccount.html",
  "lifestyle-bags",
  "brand",
  "lifestyle-bags",
  "lifestyle-drinkware",
  "registersuccess.html",
  "apparel-womens",
  "new",
  "revieworder.html",
  "lifestyle",
  "ordercompleted.html",
  "brand-youtube",
  "home-apparel-hats--quickview",
  "google+redesign-waze",
  "apparel-unisex",
  "apparel-kids",
  "brand-google",
  "lifestyle-drinkware",
  "home-apparel--quickview",
  "lifestyle-small",
  "stationery-notebooks",
  "campus",
  "lifestyle-bags",
  "google+redesign-apparel-android+tone+hoodie+black",
  "google+redesign-apparel-google+tee+white",
  "backpacks--quickview",
  "brand-android",
  "store-policies-frequently-asked-questions-",
  "stationery-stickers",
  "google+redesign-accessories-fun",
  "identifydiscount.html",
  "google+redesign-office",
  "lifestyle-drinkware",
  "apparel-unisex",
  "google+redesign-apparel-google+black+tee",
  "stationery-writing",
  "apparel-socks",
  "stationery",
  "lifestyle-bags",
  "lifestyle-drinkware",
  "google+redesign-apparel-google+zip+hoodie+fc",
  "apparel-womens",
  "apparel-unisex",
  "mugs & tumblers--quickview",
  "store-policies-shipping-information-",
  "office--quickview",
  "apparel-womens",
  "google+redesign-accessories-google+see+no+hear+no+set",
  "lifestyle-drinkware",
  "google+redesign-apparel-google+sherpa+zip+hoodie+charcoal",
  "campus",
  "google+redesign-accessories",
  "wishlist.html",
  "lifestyle-drinkware",
  "google+redesign-apparel-google+zip+hoodie+black",
  "google+redesign-apparel-google+fc+longsleeve+ash",
  "special-request-form-",
  "google+redesign-office-google+soft+modal+scarf",
  "google+redesign-electronics",
  "apparel-womens",
  "google+redesign-apparel-google+mountain+view+tee+blue",
  "google+redesign-accessories-google+lovehandle+black",
  "lifestyle-bags",
  "google+redesign-apparel-google+mens+puff+jacket+black",
  "google+redesign-apparel-google+fc+longsleeve+charcoal",
  "google+redesign-accessories-noogler+android+figure+2019",
  "lifestyle-drinkware",
  "apparel-unisex",
];
