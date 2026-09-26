class HealerManager {
    constructor() {
        this.spawnRate = 15;
        this.spacing = (width / 4);
        this.healers = [];
        this.destructionParticles = [];

    }
    show() {
        for (let healer of this.healers) {
            healer.show();
        }
        for (let particle of this.destructionParticles) {
            particle.show();
        }
    }
    update(blockManager, game) {
        if (frameCount % (this.spawnRate * 60) == 0 && game.startSpawning) {
            this.spawnHealer(blockManager);
        }
        for (let i = this.healers.length - 1; i >= 0; i--) {
            this.healers[i].update();
        }
        for (let particle of this.destructionParticles) {
            particle.update();
        }
        for (let i = this.destructionParticles.length - 1; i >= 0; i--) {
            if (this.destructionParticles[i].dead || this.destructionParticles[i].outOfBounds()) {
                this.destructionParticles.splice(i, 1);
            }
        }
    }
    spawnHealer(blockManager) {
        let x;
        let y;
        let rIndex;
        let added = false;
        let count = 0;
        while (true) {
            rIndex = floor(random(0, 4));
            x = rIndex * this.spacing + this.spacing * 0.5;
            y = random(-100, -50);
            let allOk = true;
            for (let block of blockManager.blocks) {
                let bIndex = floor(block.x / this.spacing);

                if (bIndex == rIndex) {
                    let yDiff = abs(block.y - y);
                    //console.log(yDiff);
                    if (yDiff < this.spacing * 2) {
                        allOk = false;
                    }
                }
            }
            if (allOk) {
                added = true;
                break;
            }
            count++;
            if (count > 200) {
                break;
            }
        }
        if (added) {
            this.healers.push(new Healer(x, y, this.spacing * 0.5, this.spacing * 0.5, (rIndex < 2) ? 0 : 255));
        }
    }
    checkCollisions(player) {
        for (let i = this.healers.length - 1; i >= 0; i--) {
            let healer = this.healers[i];
            if (healer.y > height + healer.w) {
                this.healers.splice(i, 1);
            } else {
                if (healer.checkCollision(player.leftCar) ||
                    healer.checkCollision(player.rightCar)) {
                    let numDestructionParticles = 40;
                    let destructionR = 10;
                    for (let i = 0; i < numDestructionParticles; i++) {
                        let a = map(i, 0, numDestructionParticles, 0, TWO_PI);
                        let px = healer.x + destructionR * cos(a);
                        let py = healer.y + destructionR * sin(a);
                        let psize = 4;
                        this.destructionParticles.push(new Particle(px, py, random(3, 7), a,
                            color(0, 255, 0, 255), psize, random(5, 8)));

                    }
                    this.healers.splice(i, 1);
                }
            }
        }
    }
}