# TensorFlow.js Deployment Example : Browser Extension

This example creates a Chrome extension (v3), enabling users to right-click on
images within a web page, and perform multi-class object detection on them. The
extension will apply a MobileNetV2 classifier to the image, and then print
the predicted class on top of the image.

To build the extension, use the command:

```sh
yarn
yarn build
```

To install the unpacked extension in chrome, follow the [instructions here](https://developer.chrome.com/extensions/getstarted).  Briefly, navigate to `chrome://extensions`, make sure that the `Developer mode` switch is turned on in the upper right, and click `Load Unpacked`.  Then select the appropriate directory (the `dist` directory containing `manifest.json`);

If it worked you should see an icon for the `TF.js mobilenet` Chrome extension.

![install page illustration](./install.png "install page")


Using the extension
----
Once the extension is installed, you should be able to classify images in the browser.  To do so, navigate to a site with images on it, such as the Google image search page for the term "tiger" used here.  Then right click on the image you wish to classify.  You should see a menu option for `Classify image with TensorFlow.js`.  Clicking that image should cause the extension to execute the model on the image, and then add some text over the image indicating the prediction.

![usage](./usage.png "usage")


Removing the extension
----
To remove the extension, click `Remove` on the extension page, or use the `Remove from Chrome...` menu option when right clicking the icon.

## Development notes

Here is how the extension works at a high level:

- A [service worker](https://developer.chrome.com/docs/extensions/mv3/migrating_to_service_workers/) `src/service_worker.js` is created that bundles
the TFJS union package and the [mobilenet model](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet). We use a bundler
([parcel](https://parceljs.org/)) to bundle everything together so no external
scripts are loaded at runtime. This is to comply with the requirement of Chrome Extension V3. Note that a service worker can still load external resources
(such as TFJS models).
- Create a context menu item in the service worker that operates on images.
When the menu item is clicked, the image src is sent to the content script for
processing. Note that in a service worker, DOM objects (e.g. Image, document,
etc) are not available.
- After the content script `content.js` receives the image src, it loads the
image, renders the image on an `OffscreenCanvas`, gets the `ImageData` from the
canvas, and sends the data back to the service worker.
- After the service worker receives the image data, it runs the mobilenet model
with the data, and gets the prediction results. It then sends to results back
to the content script for display.
- After the content script receives the results, it overlays the results on top
of the original image.
