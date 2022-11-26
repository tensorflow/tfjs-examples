"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
exports.__esModule = true;
exports.HardTanh = exports.Activation = exports.hardTanh = void 0;
/// <amd-module name="hardTanh" />
var tfc = require("@tensorflow/tfjs-core");
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("@tensorflow/tfjs-layers");
console.log(tfl);
function hardTanh(x) {
    return (0, tfjs_core_1.tidy)(function () {
        var y = tfc.mul(.5, x);
        return tfc.clipByValue(y, -1, 1);
    });
}
exports.hardTanh = hardTanh;
var Activation = /** @class */ (function (_super) {
    __extends(Activation, _super);
    function Activation() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Activation.prototype.getConfig = function () {
        return {};
    };
    return Activation;
}(tfjs_core_1.serialization.Serializable));
exports.Activation = Activation;
/**
 * Segment-wise linear approximation of tanh.
 */
var HardTanh = /** @class */ (function (_super) {
    __extends(HardTanh, _super);
    function HardTanh() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    HardTanh.prototype.apply = function (x) {
        return hardTanh(x);
    };
    /** @nocollapse */
    HardTanh.className = 'hardTanh';
    return HardTanh;
}(Activation));
exports.HardTanh = HardTanh;
tfjs_core_1.serialization.registerClass(HardTanh);
