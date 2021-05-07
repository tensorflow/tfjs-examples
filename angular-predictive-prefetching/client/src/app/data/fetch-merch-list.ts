import { Observable, of } from 'rxjs';
import { MERCH } from './demo-merch';
import { Merch } from './merch';

export const getMerchList = (category?: string): Observable<Merch[]> => {
  if (category) {
    return of(
      MERCH.filter((merch) => merch.category.startsWith(category))
    );
  }
  return of(MERCH);
};

export const getUrl = (merch: Merch): string => {
  if (!merch) return undefined;
  return `https://firebasestorage.googleapis.com/v0/b/merch-store-daa40.appspot.com/o/${merch.id}.webp?alt=media`;
};
