let game = null;
let video;
let poseNet;
let pose;
let skeleton;
let brain;
let poseLabelLeft = "None";
let poseLabelRight = "None";
let poseNetBrainReady = false;
let poseNetReady = false;
let gameFont;
let startClassifying = false;
let backgroundMusic;
let SPEED = 8;
let whooshSound;
let volumeImg;
let muteVolumeImg;
function preload() {
    gameFont = loadFont("fonts/Architects_Daughter/ArchitectsDaughter-Regular.ttf");
    backgroundMusic = loadSound("sounds/background.mp3");
}
function setup() {
    createCanvas(300, 800);
    video = createCapture(VIDEO);
    video.hide();
    poseNet = ml5.poseNet(video, modelLoaded);
    poseNet.on("pose", gotPoses);
    let options = {
        inputs: 2,
        outputs: 2,
        task: 'classification',
        debug: true
    };
    brain = ml5.neuralNetwork(options);
    const modelInfo = {
        model: 'model2/model.json',
        metadata: 'model2/model_meta.json',
        weights: 'model2/model.weights.bin',
    };
    brain.load(modelInfo, brainLoaded);
    //backgroundMusic.loop();

}

function brainLoaded() {
    console.log("Pose net brain loaded");
    poseNetBrainReady = true;

}


function gotResult(error, results) {
    if (game.paused) {
        return;
    }
    if (results[0].confidence > 0.3) {
        poseLabelLeft = results[0].label.toUpperCase();
        game.moveLeftCar(poseLabelLeft);

    }
    classifyPose();
}



function classifyPose() {
    if (game.paused) {
        return;
    }
    if (pose) {
        let inputs = [pose.leftWrist.x - pose.leftElbow.x, pose.leftWrist.y - pose.leftElbow.y];
        brain.classify(inputs, gotResult);
        let inputsRight = [pose.rightWrist.x - pose.rightElbow.x, pose.rightWrist.y - pose.rightElbow.y];
        brain.classify(inputsRight, function (error, results) {
            poseLabelRight = results[0].label.toUpperCase();
            game.moveRightCar(poseLabelRight);

        });
    } else {
        setTimeout(classifyPose, 100);
    }
}



function modelLoaded() {
    console.log("Posenet ready");
    poseNetReady = true;
}


function gotPoses(result) {
    if (result.length > 0) {
        // if (result[0].pose.score < 0.2) {
        //     game.paused = true;
        //     game.backgroundAlpha = 200;
        // }
        // else {
        //     game.paused = false;
        //     game.disappearBackground = true;
        // }
        pose = result[0].pose;
        skeleton = result[0].skeleton;

    }
}


function draw() {
    background(0);
    if (poseNetBrainReady && poseNetReady && game == null) {
        game = new Game();
    }
    if (game !== null) {
        rectMode(CORNER);
        noStroke();
        fill(255);
        rect(0, 0, width / 2, height);
        fill(0);
        rect(width / 2, 0, width / 2, height);
        noFill();
        strokeWeight(10);
        stroke(0);
        line(0, 0, 0, height);
        line(0, 0, width / 2, 0);
        line(0, height, width / 2, height);
        stroke(255);
        line(width - 3, 0, width - 3, height);
        line(width / 2, 0, width, 0);
        line(width / 2, height, width, height);
        game.render();
        game.update(mouseX, mouseY);
        fill(255);
        noStroke();
        textSize(20);
        //text(poseLabelLeft, width * 0.1, height * 0.9);
        //text(poseLabelRight, width * 0.8, height * 0.9);
    } else {
        fill(255);
        textFont(gameFont);
        textSize(50);
        text("Loading.....", width * 0.2, height * 0.5);
    }
    // fill(255, 50);
    // stroke(255);
    // strokeWeight(4);
    // rect(width * 0.80, height * 0.9, 50, 50);
    // image(volumeImg, width * 0.80, height * 0.9, 50, 50);
}

function mousePressed() {
    if (game !== null) {
        game.checkMouseEvents(mouseX, mouseY);
    }
}

function keyPressed() {
    if (game !== null) {
        game.checkKeyboardEvents(key, keyCode);
    }

}