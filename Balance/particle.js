class Particle {
    constructor(x, y, speed, angle, initialColor, initialSize,
        initialLifetime) {
        this.x = x;
        this.y = y;
        this.speed = speed;
        this.angle = angle;
        this.size = initialSize;
        this.color = initialColor;
        this.initalLifeTime = initialLifetime;
        this.lifetime = initialLifetime;
        this.dead = false;
    }

    update() {
        if (this.lifetime > 0) {
            //compute displacements
            let dx = this.speed * cos(this.angle);
            let dy = -this.speed * sin(this.angle);
            //update the position
            this.x += dx;
            this.y += dy;
            this.lifetime -= 0.1;
        } else {
            this.dead = true;
        }
    }

    outOfBounds() {
        return (this.x < -this.size || this.x > width + this.size || this.y < -this.size || this.y > height + this.size);
    }

    show() {
        noStroke();
        let c = (this.x < width * 0.5) ? 0 : 255;
        let al = map(this.lifetime, 0, this.initalLifeTime, 50, 200);
        fill(c, al);
        ellipse(this.x, this.y, this.size * 2);
    }
}


