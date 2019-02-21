/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {TextData} from './data';

// tslint:disable:max-line-length
const FAKE_TEXT = `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tempor aliquet justo non varius. Curabitur eget convallis velit. Vivamus malesuada, tortor ut finibus posuere, libero lacus eleifend felis, sit amet tempus dolor magna id nibh. Praesent non turpis libero. Praesent luctus, neque vitae suscipit suscipit, arcu neque aliquam justo, eget gravida diam augue nec lorem. Etiam scelerisque vel nibh sit amet maximus. Praesent et dui quis elit bibendum elementum a eget velit. Mauris porta lorem ac porttitor congue. Vestibulum lobortis ultrices velit, vitae condimentum elit ultrices a. Vivamus rutrum ultrices eros ac finibus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Morbi a purus a nibh eleifend convallis. Praesent non turpis volutpat, imperdiet lacus in, cursus tellus. Etiam elit velit, ornare sit amet nulla vel, aliquam iaculis mauris.

Phasellus sed sem ut justo sollicitudin cursus at sed neque. Proin tempor finibus nisl, nec aliquam leo porta at. Nullam vel mauris et neque pellentesque laoreet sit amet eu risus. Sed sed ante sed enim hendrerit commodo. Etiam blandit aliquet molestie. Nullam dictum imperdiet enim, quis scelerisque nunc ultricies sit amet. Praesent dictum dictum lobortis. Sed ut ipsum at orci commodo congue.

Aenean pharetra mollis erat, id convallis ante elementum at. Cras semper turpis nec lorem tempus ultrices. Sed eget purus vel est blandit dictum. Praesent auctor, sapien non consequat pellentesque, risus orci sagittis leo, at cursus nibh nisi vel quam. Morbi et orci id quam dictum efficitur ac iaculis nisl. Donec at nunc et nibh accumsan malesuada eu in odio. Donec quis elementum turpis. Vestibulum pretium rhoncus orci, nec gravida nisl hendrerit pellentesque. Cras imperdiet odio a quam mollis, in aliquet neque efficitur. Praesent at tincidunt ipsum. Maecenas neque risus, pretium ut orci sit amet, dignissim auctor dui. Sed finibus nunc elit, rhoncus ornare dui pharetra vitae. Sed ut iaculis ex. Quisque quis molestie ligula. Vivamus egestas rhoncus mollis.

Pellentesque volutpat ipsum vitae ex interdum, eu rhoncus dolor fringilla. Suspendisse potenti. Maecenas in sem leo. Curabitur vestibulum porta vulputate. Nunc quis consectetur enim. Aliquam congue, augue in commodo porttitor, sem tellus posuere augue, ut aliquam sapien massa in est. Duis convallis pellentesque vehicula. Mauris ipsum urna, congue consequat posuere sed, euismod nec mauris. Praesent sollicitudin scelerisque scelerisque. Ut commodo nisl vitae nunc feugiat auctor. Praesent imperdiet magna facilisis nunc vulputate, vel suscipit leo consequat. Duis fermentum rutrum ipsum a laoreet. Nunc dictum libero in quam pellentesque, sit amet tempus tellus suscipit. Curabitur pharetra erat bibendum malesuada rhoncus.

Donec laoreet leo ligula, ut condimentum mi placerat ut. Sed pretium sollicitudin nisl quis tincidunt. Proin id nisl ornare, interdum lorem quis, posuere lacus. Cras cursus mollis scelerisque. Mauris mattis mi sed orci feugiat, et blandit velit tincidunt. Donec ultrices leo vel tellus tincidunt, id vehicula mi commodo. Nulla egestas mollis massa. Etiam blandit nisl eu risus luctus viverra. Mauris eget mi sem.

`;
// tslint:enable:max-line-length

describe('TextData', () => {
  it('Creation', () => {
    const data = new TextData('LoremIpsum', FAKE_TEXT, 20, 3);
    expect(data.sampleLen()).toEqual(20);
    expect(data.charSetSize()).toBeGreaterThan(0);
  });

  it('nextDataEpoch: full pass', () => {
    const data = new TextData('LoremIpsum', FAKE_TEXT, 20, 3);
    const [xs, ys] = data.nextDataEpoch();
    expect(xs.rank).toEqual(3);
    expect(ys.rank).toEqual(2);
    expect(xs.shape[0]).toEqual(ys.shape[0]);
    expect(xs.shape[1]).toEqual(20);
    expect(xs.shape[2]).toEqual(ys.shape[1]);
  });

  it('nextDataEpoch: partial pass', () => {
    const data = new TextData('LoremIpsum', FAKE_TEXT, 20, 3);
    const [xs, ys] = data.nextDataEpoch(4);
    expect(xs.rank).toEqual(3);
    expect(ys.rank).toEqual(2);
    expect(xs.shape[0]).toEqual(4);
    expect(ys.shape[0]).toEqual(4);
    expect(xs.shape[1]).toEqual(20);
    expect(xs.shape[2]).toEqual(ys.shape[1]);
  });

  it('getFromCharSet', () => {
    const data = new TextData('LoremIpsum', FAKE_TEXT, 20, 3);
    const charSetSize = data.charSetSize();
    expect(data.getFromCharSet(0)).not.toEqual(data.getFromCharSet(1));
    expect(data.getFromCharSet(0))
        .not.toEqual(data.getFromCharSet(charSetSize - 1));
    expect(data.getFromCharSet(charSetSize)).toBeUndefined();
    expect(data.getFromCharSet(-1)).toBeUndefined();
  });

  it('getRandomSlice', () => {
    const data = new TextData('LoremIpsum', FAKE_TEXT, 20, 3);
    const [text, indices] = data.getRandomSlice();
    expect(typeof text).toEqual('string');
    expect(Array.isArray(indices)).toEqual(true);
  });
});

