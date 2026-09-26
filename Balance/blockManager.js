class BlockManager {
    constructor(playerXPositions) {
        this.blocks = [];
        this.leftLaneY = -50;
        this.rightLaneY = -100;
        this.spacing = (width * 0.5) / 2;
        this.spawnRate = 3; //in seconds
        this.rightLaneXPositions = [];
        this.leftLaneXPositions = [];
        this.destructionParticles = [];
        for (let i = 0; i < 2; i++) {
            this.leftLaneXPositions[i] = playerXPositions[i];
        }
        for (let i = 2; i < 4; i++) {
            this.rightLaneXPositions[i % 2] = playerXPositions[i];
        }
    }

    spawnBlocks() {
        let leftLaneFillArr = getFillArr(1);
        let rightLaneFillArr = getFillArr(1);
        //leftlanespawn 
        for (let i = 0; i < leftLaneFillArr.length; i++) {
            if (leftLaneFillArr[i] == 1) {
                this.blocks.push(new Block(this.leftLaneXPositions[i], this.leftLaneY, this.spacing * 0.5, this.spacing * 0.5, 0));
            }
        }
        for (let i = 0; i < rightLaneFillArr.length; i++) {
            if (rightLaneFillArr[i] == 1) {
                this.blocks.push(new Block(this.rightLaneXPositions[i], this.rightLaneY, this.spacing * 0.5, this.spacing * 0.5, 255));
            }
        }

    }
    show() {
        for (let block of this.blocks) {
            block.show();
        }
        for (let particle of this.destructionParticles) {
            particle.show();
        }
    }
    update(game) {
        for (let block of this.blocks) {
            block.update();
        }
        for (let particle of this.destructionParticles) {
            particle.update();
        }
        for (let i = this.destructionParticles.length - 1; i >= 0; i--) {
            if (this.destructionParticles[i].dead || this.destructionParticles[i].outOfBounds()) {
                this.destructionParticles.splice(i, 1);
            }
        }
        if (frameCount % (60 * this.spawnRate) == 0 && game.startSpawning) {
            this.spawnBlocks();

        }

    }
    checkCollisions(player) {
        for (let i = this.blocks.length - 1; i >= 0; i--) {
            let block = this.blocks[i];
            if (block.y > height + block.w) {
                this.blocks.splice(i, 1);

            } else {
                if (block.checkCollision(player.leftCar) ||
                    block.checkCollision(player.rightCar)) {
                    let numDestructionParticles = 40;
                    let destructionR = 10;
                    for (let i = 0; i < numDestructionParticles; i++) {
                        let a = map(i, 0, numDestructionParticles, 0, TWO_PI);
                        let px = block.x + destructionR * cos(a);
                        let py = block.y + destructionR * sin(a);
                        let psize = 4;
                        this.destructionParticles.push(new Particle(px, py, random(3, 7), a,
                            color(0, 255, 0, 255), psize, random(5, 8)));

                    }
                    this.blocks.splice(i, 1);
                }
            }
        }
    }
}

function getFillArr(atMost) {
    let num = floor(random(1, atMost + 1));
    let arr = new Array(atMost + 1);
    for (let i = 0; i < arr.length; i++) {
        arr[i] = 0;
    }
    for (let i = 0; i < num; i++) {
        let index = floor(random(0, arr.length));
        arr[index] = 1;
    }
    return arr;
}