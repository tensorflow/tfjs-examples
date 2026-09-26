class Player {
    constructor(playerXPositions) {
        this.playerXPositions = playerXPositions;
        let a = this.playerXPositions.slice(0, 2);
        let b = this.playerXPositions.slice(2, 4);
        this.leftCar = new Car("left", a, height * 0.8, 0);
        this.rightCar = new Car("right", b, height * 0.8, 255);
    }
    show() {
        this.leftCar.show();
        this.rightCar.show();
    }
    update() {
        this.leftCar.update();
        this.rightCar.update();
    }
    playerMoveLeftCar(moveDir) {
        this.leftCar.move(moveDir);
    }
    playerMoveRightCar(moveDir) {
        this.rightCar.move(moveDir);
    }
}