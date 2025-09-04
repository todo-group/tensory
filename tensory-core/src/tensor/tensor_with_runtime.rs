use crate::tensor::{Tensor, TensorBroker, TensorRepr};

pub struct TensorWithRuntime<'rt, B: TensorBroker, R: TensorRepr, RT> {
    tensor: Tensor<B, R>,
    runtime: &'rt RT,
}

impl<B: TensorBroker, R: TensorRepr> Tensor<B, R> {
    pub fn bind<'rt, RT>(self, runtime: &'rt RT) -> TensorWithRuntime<'rt, B, R, RT> {
        TensorWithRuntime::from_raw(self, runtime)
    }
}
impl<'rt, B: TensorBroker, R: TensorRepr, RT> TensorWithRuntime<'rt, B, R, RT> {
    pub fn from_raw(tensor: Tensor<B, R>, runtime: &'rt RT) -> Self {
        Self { tensor, runtime }
    }
    pub fn into_raw(self) -> (Tensor<B, R>, &'rt RT) {
        (self.tensor, self.runtime)
    }
    pub fn unbind(self) -> Tensor<B, R> {
        self.tensor
    }
    pub fn runtime(&self) -> &'rt RT {
        self.runtime
    }
}

// pub enum ContractExecutorError<Err> {
//     MismatchedExecutors,
//     Internal(Err),
// }

// trait Executor {
//     fn exec_eq(&self, other: &Self) -> bool;
// }

// impl<Id, L: TensorRepr, Ex: Executor> Tensor<Id, L, Ex> {
//     pub fn contract<R: TensorRepr>(self,rhs: Tensor<Id, R, Ex>) -> Result<Tensor<Id, Ex::Res, Ex>, ContractExecutorError<Ex::Err>> where Ex: ContractionContext<L, R> {
//         self.executor.exec_eq(&rhs.executor).then(()).ok_or(ContractExecutorError::MismatchedExecutors).and()

//         //map(|_|TensorMul::new(self, rhs).by(context).map_err(ContractExecutorError::Internal))
//         todo!()
//     }
// }

// impl<Id, L: TensorRepr, R: TensorRepr, Ex: ContractionContext<L, R>> Mul<Tensor<Id, R, Ex>>
//     for Tensor<Id, L, Ex>
// {
//     type Output = TensorMul<Id, <Ex as ContractionContext<L, R>>::Res, Ex>;

//     fn mul(self, rhs: Tensor<Id, R, Ex>) -> Self::Output {
//         if self.executor != rhs.executor {
//             panic!("Cannot multiply tensors with different executors");
//         }

//         TensorMul::new(self, rhs)
//     }
// }

// impl<Id, A: TensorRepr, Ex: SvdContext<A>> Tensor<Id, A, Ex> {
//     pub fn svd<'a>(self, u_legs: LegRefSet<'a, Id>) ->  {
//         TensorSvd::new(self, u_legs)
//     }
// }
